JD_EXTRACTION_PROMPT = """
Extract the following from the job description:

Return valid JSON with:
- required_skills
- preferred_skills
- responsibilities
- ats_keywords
- seniority
"""

RESUME_GENERATION_PROMPT = """
You are a resume editor.

Rules:
- You may ONLY rephrase or combine the provided bullet points
- Do NOT introduce new technologies or accomplishments
- Preserve metrics exactly
- Optimize wording to match the job description language
- Keep bullets concise and ATS-friendly
- Order bullets by relevance to the job requirements
- Return them in JSON format under the object name "rewritten_bullets"
- Generate 5-15 Skills that match job requirements under the same restraints as above
- These skills will target ATS keywords that were not covered under the rewritten bullets
- Do NOT mention specific frameworks or regulations unless explicitly contained in the original bullets
- Return the skills in JSON format under the object name "targeted_skills"

Job Requirements:
{job_requirements}

Source Bullets:
{bullets}

Generate tailored resume bullets and skills.
"""

CRITIC_PROMPT = """
Review the generated resume bullets.

Return JSON with:
- unsupported_claims
- missing_required_skills
- generic_bullets
"""