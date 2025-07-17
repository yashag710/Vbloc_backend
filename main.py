print("!!!!!!!!!! SERVER IS STARTING THIS FILE !!!!!!!!!!!")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

load_dotenv()

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client using OpenAI-compatible syntax
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Request models
class ConceptRequest(BaseModel):
    video_id: str
    transcript: str

class QnARequest(BaseModel):
    video_id: str
    transcript: str

class FlashcardRequest(BaseModel):
    video_id: str
    transcript: str

class TopicStructureRequest(BaseModel):
    video_id: str
    transcript: str

    
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    
class VideoChatRequest(BaseModel):
    video_id: str
    transcript: str
    new_question: str
    chat_history: List[ChatMessage]

# Route for the chatbot   
@app.post("/video-chat")
def video_chat(req: VideoChatRequest):
    try:
        # Limit transcript to avoid exceeding model token limit
        trimmed_transcript = req.transcript[:4000]

        # Build system prompt
        system_prompt = f"""
You are an AI assistant that helps users understand a YouTube video.
Use the following transcript to answer their questions.

Transcript:
{trimmed_transcript}
"""

        # Construct messages with chat history
        chat_messages = [{"role": "system", "content": system_prompt}]
        for msg in req.chat_history:
            chat_messages.append({"role": msg.role, "content": msg.content})
        chat_messages.append({"role": "user", "content": req.new_question})

        # Call Groq model
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=chat_messages,
            temperature=0.5,
            max_tokens=1000
        )

        assistant_reply = response.choices[0].message.content.strip()

        return {
            "new_chat_turn": {
                "role": "assistant",
                "content": assistant_reply
            }
        }

    except Exception as e:
        print(f"Error in /video-chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# GET transcript (no caching as frontend handles it)
@app.get("/transcript/{video_id}")
def get_transcript(video_id: str):
    try:
        print(f"Fetching transcript for video ID: {video_id}")
        
        # Try to get transcript with more detailed error handling
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as transcript_error:
            print(f"Transcript API error: {str(transcript_error)}")
            
            # Try to get transcript in different languages
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                # Try to find any available transcript
                for transcript_info in transcript_list:
                    try:
                        transcript = transcript_info.fetch()
                        break
                    except:
                        continue
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"No transcripts available for video {video_id}. Original error: {str(transcript_error)}"
                    )
            except Exception as list_error:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cannot access video {video_id}. It may be private, age-restricted, or have no captions. Error: {str(list_error)}"
                )
        
        # Process transcript
        if not transcript:
            raise HTTPException(status_code=400, detail="Empty transcript received")
            
        full_text = " ".join([entry["text"] for entry in transcript])
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Transcript is empty")
            
        print(f"Successfully fetched transcript. Length: {len(full_text)} characters")
        return {"video_id": video_id, "transcript": full_text}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Check video endpoint
@app.get("/check-video/{video_id}")
def check_video(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_transcripts = []
        
        for transcript in transcript_list:
            available_transcripts.append({
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            })
        
        return {
            "video_id": video_id,
            "video_accessible": True,
            "available_transcripts": available_transcripts
        }
    except Exception as e:
        return {
            "video_id": video_id,
            "video_accessible": False,
            "error": str(e)
        }

def extract_json_from_llm_response(text: str):
    """Extract JSON array from LLM response that might contain extra text"""
    print(f"Processing LLM response: {text[:500]}...")
    
    # Clean the text first
    text = text.strip()
    
    # Remove common markdown wrappers
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    
    # Remove markdown headers and bold text
    text = re.sub(r'^#+\s*.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*[^*]*\*\*', '', text)
    text = re.sub(r'\*[^*]*\*', '', text)
    
    # Find JSON array patterns - more comprehensive search
    json_patterns = [
        r'(\[\s*\{[^}]*\}(?:\s*,\s*\{[^}]*\})*\s*\])',  # Basic array pattern
        r'(\[(?:[^[\]]*\{[^}]*\}[^[\]]*)*\])',  # Array with content before/after
        r'(\[\s*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}(?:\s*,\s*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\})*\s*\])',  # Nested objects
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            json_candidate = match.group(1)
            try:
                # Clean up the JSON candidate
                json_candidate = json_candidate.strip()
                parsed = json.loads(json_candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    print(f"Successfully parsed JSON with {len(parsed)} items")
                    return parsed
            except json.JSONDecodeError as e:
                print(f"JSON decode error for pattern: {e}")
                continue
    
    # Try to find array boundaries manually
    brace_count = 0
    bracket_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '[':
            if start_idx == -1:
                start_idx = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0 and start_idx != -1:
                json_candidate = text[start_idx:i + 1]
                try:
                    parsed = json.loads(json_candidate)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        print(f"Successfully parsed JSON using manual boundary detection")
                        return parsed
                except json.JSONDecodeError:
                    pass
        elif char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
    
    # Last resort: try to parse the entire text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # If all else fails, return a formatted error
    print(f"Failed to extract JSON from: {text[:300]}...")
    raise ValueError(f"Could not extract valid JSON array from LLM response. Response started with: {text[:200]}...")

# Enhanced extract topic structure endpoint
@app.post("/extract-topic-structure")
def extract_topic_structure(payload: TopicStructureRequest):
    try:
        topic_prompt = f"""
CRITICAL: You must respond with ONLY a valid JSON array. No markdown, no explanations, no extra text.

Extract the main topics and subtopics from this transcript and format as JSON array:

[
  {{
    "topic": "Topic Name",
    "subtopics": ["Subtopic 1", "Subtopic 2", "Subtopic 3"]
  }},
  {{
    "topic": "Another Topic",
    "subtopics": ["Sub A", "Sub B"]
  }}
]

Transcript: {payload.transcript[:4000]}

Response (JSON array only):
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a JSON generator. You ONLY respond with valid JSON arrays. No explanations, no markdown, no extra text."},
                {"role": "user", "content": topic_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        raw_output = response.choices[0].message.content.strip()
        print(f"Raw topic structure output: {raw_output}")

        try:
            parsed_json = extract_json_from_llm_response(raw_output)
            print(f"Parsed topics: {parsed_json}")
            return {"topics": parsed_json}
        except Exception as parse_error:
            print(f"Topic structure JSON parsing failed: {parse_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse topic structure. Error: {str(parse_error)}"
            )

    except Exception as e:
        print(f"Error in extract_topic_structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced extract concepts endpoint with comprehensive details
@app.post("/extract-concepts")
def extract_concepts(req: ConceptRequest):
    try:
        concept_prompt = f"""
CRITICAL: You must respond with ONLY a valid JSON array. No markdown, no explanations, no extra text.

Extract 4-6 key concepts from this transcript and format as JSON:

[
  {{
    "concept": "Concept Name",
    "definition": "Clear definition in 2-3 sentences",
    "detailed_explanation": "Step-by-step explanation in 4-5 sentences",
    "formulas": ["Formula 1", "Formula 2"],
    "real_world_examples": ["Example 1", "Example 2", "Example 3"],
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "common_mistakes": ["Mistake 1", "Mistake 2"],
    "study_tips": ["Tip 1", "Tip 2"],
    "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
    "applications": ["Application 1", "Application 2"],
    "difficulty_level": "Beginner/Intermediate/Advanced",
    "estimated_study_time": "Time estimate",
    "practice_problems": ["Problem 1", "Problem 2"],
    "additional_notes": "Extra insights"
  }}
]

Transcript: {req.transcript[:4000]}

Response (JSON array only):
"""

        concept_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a JSON generator. You ONLY respond with valid JSON arrays. No explanations, no markdown, no extra text."},
                {"role": "user", "content": concept_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )

        raw_content = concept_response.choices[0].message.content.strip()
        print(f"Raw concept response: {raw_content}")

        try:
            concept_list = extract_json_from_llm_response(raw_content)
            print(f"Parsed concepts: {concept_list}")
            
            return {"video_id": req.video_id, "concepts": concept_list}
            
        except Exception as parse_error:
            print(f"Concept JSON parsing failed: {parse_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse concepts. Error: {str(parse_error)}"
            )

    except Exception as e:
        print(f"Error in extract_concepts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_json_from_llm_response2(response_text):
    """
    Extract JSON from LLM response, handling various formats and edge cases
    """
    try:
        # Remove any markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        # Remove any leading/trailing whitespace
        response_text = response_text.strip()
        
        # Find JSON array boundaries
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON array found in response")
        
        json_str = response_text[start_idx:end_idx + 1]
        
        # Try to parse the JSON
        parsed_json = json.loads(json_str)
        
        # Validate that it's a list
        if not isinstance(parsed_json, list):
            raise ValueError("Response is not a JSON array")
        
        # Validate each question object
        for i, question in enumerate(parsed_json):
            if not isinstance(question, dict):
                raise ValueError(f"Question {i} is not a valid object")
            
            # Required fields
            required_fields = ['question', 'type', 'explanation', 'difficulty', 'topic']
            for field in required_fields:
                if field not in question:
                    raise ValueError(f"Question {i} missing required field: {field}")
            
            # Type-specific validation
            if question['type'] == 'mcq':
                if 'options' not in question or 'correct_answer' not in question:
                    raise ValueError(f"MCQ question {i} missing options or correct_answer")
            elif question['type'] == 'numerical':
                if 'expected_answer' not in question or 'solution_steps' not in question:
                    raise ValueError(f"Numerical question {i} missing expected_answer or solution_steps")
            elif question['type'] == 'output':
                if 'expected_answer' not in question:
                    raise ValueError(f"Output question {i} missing expected_answer")
        
        return parsed_json
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic JSON: {response_text}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"General parsing error: {e}")
        print(f"Response text: {response_text}")
        raise ValueError(f"Failed to parse response: {str(e)}")

@app.post("/generate-qna")
def generate_qna(req: QnARequest):
    try:
        # Limit transcript length to prevent token overflow
        transcript_chunk = req.transcript[:3000]  # Reduced from 4000
        
        prompt = f"""
CRITICAL: You must respond with ONLY a valid JSON array. No markdown, no explanations, no extra text.

Generate exactly 6 questions from this transcript. Mix question types (mcq, numerical, output).

FORMAT REQUIREMENTS:
- MCQ questions need: question, type, options, correct_answer, explanation, difficulty, topic
- Numerical questions need: question, type, expected_answer, solution_steps, explanation, difficulty, topic
- Output questions need: question, type, expected_answer, explanation, difficulty, topic

EXAMPLE FORMAT:
[
  {{
    "question": "What is the main topic discussed?",
    "type": "mcq",
    "options": {{
      "A": "Option A",
      "B": "Option B", 
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "C",
    "explanation": "Explanation here",
    "difficulty": "easy",
    "topic": "Topic name"
  }},
  {{
    "question": "Calculate the value of X",
    "type": "numerical",
    "expected_answer": "42",
    "solution_steps": ["Step 1", "Step 2", "Step 3"],
    "explanation": "Solution explanation",
    "difficulty": "medium",
    "topic": "Calculations"
  }},
  {{
    "question": "What would be the output of this code?",
    "type": "output",
    "expected_answer": "Hello World",
    "explanation": "Output explanation",
    "difficulty": "easy",
    "topic": "Programming"
  }}
]

Transcript: {transcript_chunk}

Response (JSON array only):
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a JSON generator. You ONLY respond with valid JSON arrays. No explanations, no markdown, no extra text. Always complete the JSON array properly."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=4000,  # Increased token limit
            top_p=0.9
        )

        raw_output = response.choices[0].message.content.strip()
        print(f"Raw QnA response: {raw_output}")

        # Check if the response was truncated
        if not raw_output.endswith(']'):
            print("WARNING: Response appears to be truncated")
            # Try to fix incomplete JSON
            raw_output = fix_incomplete_json(raw_output)

        try:
            questions = extract_json_from_llm_response2(raw_output)
            
            # Ensure we have at least some questions
            if len(questions) == 0:
                raise ValueError("No questions generated")
            
            print(f"Successfully parsed {len(questions)} questions")
            return {"video_id": req.video_id, "questions": questions}
            
        except Exception as parse_error:
            print(f"QnA JSON parsing failed: {parse_error}")
            print(f"Raw output length: {len(raw_output)}")
            
            # Fallback: try to salvage partial questions
            try:
                partial_questions = salvage_partial_json(raw_output)
                if len(partial_questions) > 0:
                    print(f"Salvaged {len(partial_questions)} questions")
                    return {"video_id": req.video_id, "questions": partial_questions}
            except:
                pass
            
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse Q&A. Error: {str(parse_error)}"
            )

    except Exception as e:
        print(f"Error in generate_qna: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def fix_incomplete_json(json_str):
    """
    Try to fix incomplete JSON by adding missing closing brackets
    """
    try:
        # Count opening and closing brackets
        open_brackets = json_str.count('[') + json_str.count('{')
        close_brackets = json_str.count(']') + json_str.count('}')
        
        # If we're missing closing brackets, try to add them
        if open_brackets > close_brackets:
            # Find the last complete object
            last_complete = json_str.rfind('}')
            if last_complete != -1:
                # Add closing bracket for array
                json_str = json_str[:last_complete + 1] + ']'
        
        return json_str
    except:
        return json_str

def salvage_partial_json(json_str):
    """
    Try to salvage partial questions from incomplete JSON
    """
    questions = []
    
    try:
        # Find all complete question objects
        pattern = r'\{[^{}]*"question"[^{}]*\}'
        matches = re.findall(pattern, json_str, re.DOTALL)
        
        for match in matches:
            try:
                # Try to parse each individual question
                question = json.loads(match)
                # Basic validation
                if all(key in question for key in ['question', 'type', 'explanation']):
                    questions.append(question)
            except:
                continue
                
    except Exception as e:
        print(f"Error salvaging partial JSON: {e}")
    
    return questions

# New Flashcards API
@app.post("/generate-flashcards")
def generate_flashcards(req: FlashcardRequest):
    try:
        flashcard_prompt = f"""
CRITICAL: You must respond with ONLY a valid JSON array. No markdown, no explanations, no extra text.

Create 10-12 flashcards from this transcript:

[
  {{
    "id": "card_1",
    "front": "Question or concept",
    "back": "Answer with explanation",
    "category": "Formula/Definition/Concept/Application",
    "difficulty": "easy/medium/hard",
    "tags": ["tag1", "tag2"],
    "hint": "Optional hint",
    "importance": "high/medium/low"
  }}
]

Transcript: {req.transcript[:4000]}

Response (JSON array only):
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a JSON generator. You ONLY respond with valid JSON arrays. No explanations, no markdown, no extra text."},
                {"role": "user", "content": flashcard_prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )

        raw_output = response.choices[0].message.content.strip()
        print(f"Raw flashcard response: {raw_output}")

        try:
            flashcards = extract_json_from_llm_response(raw_output)
            
            # Add metadata for download
            flashcard_data = {
                "video_id": req.video_id,
                "generated_at": datetime.now().isoformat(),
                "total_cards": len(flashcards),
                "flashcards": flashcards
            }
            
            return flashcard_data
            
        except Exception as parse_error:
            print(f"Flashcard JSON parsing failed: {parse_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse flashcards. Error: {str(parse_error)}"
            )

    except Exception as e:
        print(f"Error in generate_flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Downloadable flashcards endpoint
@app.get("/download-flashcards/{video_id}")
def download_flashcards(video_id: str):
    """
    This endpoint can be used to trigger download of flashcards in different formats.
    Frontend can call this and handle the download.
    """
    return {
        "message": "Use POST /generate-flashcards to get flashcard data",
        "formats": ["json", "csv", "anki"],
        "video_id": video_id
    }




# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Enhanced error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    