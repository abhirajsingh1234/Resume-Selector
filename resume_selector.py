import os
import re
from typing import Dict, List, Any
import mammoth
from PyPDF2 import PdfReader
import json
import google.generativeai as genai
import time

genai.configure(api_key="YOUR_GEMINI_API_KEY")

class CVParser:
    """Class to parse CVs from different formats"""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def get_file_text(file_path: str) -> str:
        """Get text from file based on its extension"""
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.pdf':
            return CVParser.extract_text_from_pdf(file_path)
        elif ext.lower() == '.docx':
            return CVParser.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class GeminiScorer:
    """Class to score CVs using Gemini API"""
    
    def __init__(self, api_key: str = None):
        return None
    
    def analyze_cv(self, cv_text: str, job_description: str, job_location: str) -> dict:    

        # Create the model with appropriate configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8190,
            "response_mime_type": "text/plain",
        }

        comprehensive_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config=generation_config,
            system_instruction = """
            You are an expert recruiter analyzing a candidate's CV against a job description.
            
            Analyze the provided CV text and job details to extract the following metrics:
            
            1. LANGUAGE_QUALITY: Score the CV's language quality from 0-100 based on grammar, clarity, professional tone, terminology, and formatting.
        
            2. EXPERIENCE_YEARS: Extract the total years of professional experience from the CV. Count full-time roles only, not internships or education.
                - Use the formula: (total_candidate_experience / experience_required) * 100
            
            3. EXPERIENCE_QUALITY: Score from 0-100 how well the candidate's experience matches the job description, considering relevance of roles, technical skills, industry experience, and achievements.
            
            4. LOCATION_SCORE: Score location compatibility from 0-100:
                - 100: Same city/area as the job
                - 75-99: Within commuting distance (20km)
                - 50-74: Same region/state but may need to relocate
                - 25-49: Same country but would need to relocate
                - 0-24: International relocation required
        
            5. SKILLS_SCORE: Score from 0-100 how well the candidate's skills match the job requirements.
                -use the formula: (matched_skill/total_skill)*100
        
            Return your analysis in this exact format:
            LANGUAGE_QUALITY: [score].
            EXPERIENCE_YEARS: [years].
            EXPERIENCE_QUALITY: [score].
            LOCATION_SCORE: [score].
            SKILLS_SCORE: [score].
            """
        )
    
        try:
            start_time = time.time()
            chat_session = comprehensive_model.start_chat()
            response = chat_session.send_message(f"CV TEXT: {cv_text}\n\nJOB DESCRIPTION: {job_description}\n\nJOB LOCATION: {job_location}")
            end_time = time.time()
            time.sleep(1)
            print(f"Time taken for Gemini API call: {end_time - start_time} seconds")
            result_text = response.text.strip()
            print(result_text)
        
        # Extract all scores using regex
            language_quality = float(re.search(r'LANGUAGE_QUALITY:\s*(\d+(\.\d+)?)', result_text).group(1)) if re.search(r'LANGUAGE_QUALITY:\s*(\d+(\.\d+)?)', result_text) else 50
            experience_years = float(re.search(r'EXPERIENCE_YEARS:\s*(\d+(\.\d+)?)', result_text).group(1)) if re.search(r'EXPERIENCE_YEARS:\s*(\d+(\.\d+)?)', result_text) else 0
            experience_quality = float(re.search(r'EXPERIENCE_QUALITY:\s*(\d+(\.\d+)?)', result_text).group(1)) if re.search(r'EXPERIENCE_QUALITY:\s*(\d+(\.\d+)?)', result_text) else 50
            location_score = float(re.search(r'LOCATION_SCORE:\s*(\d+(\.\d+)?)', result_text).group(1)) if re.search(r'LOCATION_SCORE:\s*(\d+(\.\d+)?)', result_text) else 50
            skills_score = float(re.search(r'SKILLS_SCORE:\s*(\d+(\.\d+)?)', result_text).group(1)) if re.search(r'SKILLS_SCORE:\s*(\d+(\.\d+)?)', result_text) else 0
        
        # Normalize and bound scores
            language_quality = min(100, max(0, language_quality))
            experience_quality = min(100, max(0, experience_quality))
            experience_years = min(100, max(0, experience_years))
            location_score = min(100, max(0, location_score))
            skills_score = min(100, max(0, skills_score))
        
        # Normalize experience years to 0-100 scale (assuming 20+ years is max)
            years_normalized = min(experience_years / 20 * 100, 100)
        
            return {
                "language_quality": language_quality,
                "experience_years": experience_years,
                "experience_years_score": years_normalized,
                "experience_quality": experience_quality,
                "location_score": location_score,
                "skills_score": skills_score,
                "raw_response": result_text  # Including raw response for debugging
            }
        except Exception as e:
            print(f"Error in comprehensive CV analysis: {e}")
            return {
                "language_quality": 50,
                "experience_years": 0,
                "experience_years_score": 0,
                "experience_quality": 50,
                "location_score": 50,
                "skills_score": 50,
                "error": str(e)
            }


# Function to score a single candidate - moved outside the class for multiprocessing
def score_candidate(cv_path: str, job_description: str, job_location: str, skills: str, weights: Dict[str, float], api_key: str = None) -> Dict[str, Any]:
    """
    Score a candidate based on their CV and job requirements using a single Gemini API call
    Returns a dictionary with scores and overall score
    """
    try:
        cv_text = CVParser.get_file_text(cv_path)
        
        gemini_scorer = GeminiScorer(api_key)
        
        scores = gemini_scorer.analyze_cv(cv_text, job_description, job_location)
        
        weighted_score = (
            scores["language_quality"] * weights.get("language_quality", 0) +
            scores["experience_years_score"] * weights.get("experience_years", 0) +
            scores["experience_quality"] * weights.get("experience_quality", 0) +
            scores["location_score"] * weights.get("location", 0) +
            scores["skills_score"] * weights.get("skills", 0)
        ) / 100 
        
        return {
            "candidate_name": os.path.basename(cv_path),
            "language_quality": scores["language_quality"],
            "experience_years": scores["experience_years"],
            "experience_years_score": scores["experience_years_score"],
            "experience_quality": scores["experience_quality"],
            "location_score": scores["location_score"],
            "skills_score": scores["skills_score"],
            "overall_score": weighted_score,
            "cv_path": cv_path
        }
    except Exception as e:
        print(f"Error scoring candidate {cv_path}: {e}")

class RecommendationSystem:    
    def __init__(self):
        pass      
    
    def recommend_candidates(self, cv_directory: str, job_description: str, skills: str, job_location: str, 
                            weights: Dict[str, float], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Process all CVs in a directory and recommend top candidates
        Returns a list of top N candidates with their scores
        """
        cv_paths = []
        for root, _, files in os.walk(cv_directory):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx')):
                    cv_paths.append(os.path.join(root, file))
        
        print(f"Found {len(cv_paths)} CVs to process")
        
        results = []
        
        for cv_path in cv_paths:
            try:
                result = score_candidate(cv_path, job_description, job_location, skills, weights)
                results.append(result)
            except Exception as e:
                print(f"Error processing CV {cv_path}: {e}")
        
        sorted_results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
        
        return sorted_results[:top_n]
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")


def main(job_title: str, job_description: str, skills: str, job_location: str, cv_directory: str):
    
    weights = {
        "language_quality": 20,
        "experience_quality": 20,
        "skills": 25,
        "experience_years": 20,
        "location": 15
    }
    
    recommendation_system = RecommendationSystem()
    
    results = recommendation_system.recommend_candidates(
        cv_directory=cv_directory,
        job_description=job_description,
        job_location=job_location,
        skills=skills,
        weights=weights,
        top_n=10
    )

    
    output_file = f"{job_title.replace(' ', '_').lower()}_candidates.json"
    recommendation_system.save_results(results, output_file)
    
    print(f"\nTop candidates for {job_title}:")
    print("-" * 80)
    for i, candidate in enumerate(results):
        print(f"{i+1}. {candidate['candidate_name']} - Overall Score: {candidate['overall_score']:.2f}")
        print(f"   Language Quality: {candidate['language_quality']:.2f}")
        print(f"   Experience Quality: {candidate['experience_quality']:.2f}")
        print(f"   Years of Experience: {candidate['experience_years']} (Score: {candidate['experience_years_score']:.2f})")
        print(f"   Location Compatibility: {candidate['location_score']:.2f}")
        print(f"   Skills: {candidate['skills_score']:.2f}")
        print("-" * 80)


if __name__ == "__main__":
    job_title = "Senior Software Engineer"
    job_description = """
    We are looking for a Senior Software Engineer with strong experience in Python and machine learning.
    The ideal candidate has 5+ years of experience in software development, with a focus on building
    scalable data processing systems. Experience with python, machine learning, data analytics, fastapi, sql/pl-sql, flask, django, NLP, cloud platforms, and distributed computing
    is a plus. The candidate should be able to design, implement, and maintain complex software systems.
    """
    required_skills = "python, machine learning, data analytics, fastapi, sql/pl-sql, flask, django, NLP"
    job_location = "India, Mumbai"
    cv_directory = "sample_cvs"   # directory path where cv's are present
    main(job_title, job_description, required_skills, job_location, cv_directory)