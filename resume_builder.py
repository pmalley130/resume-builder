import json
from typing import List
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

import os
from dotenv import load_dotenv

from prompts import ( #import prompts from separate file
    JD_EXTRACTION_PROMPT,
    RESUME_GENERATION_PROMPT,
    CRITIC_PROMPT
)


load_dotenv() #load API key

client = OpenAI() #start openAI client

#vector store setup
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

chroma = chromadb.Client()
collection = chroma.get_or_create_collection(
    name="resume_bullets",
    embedding_function=embedding_fn
)

#load collection
def load_collection(path="data/resume_data.json"):

    with open(path,encoding='utf8') as f:
            data = json.load(f)

    #load roles
    candidate = data["candidate"]
    roles = candidate.get("roles", [])

    #build lookup table to align roles with bullets
    role_lookup = {
            (r["company"], r["title"]): r
            for r in roles
    }
    #prepare documents/bullets
    documents = []
    metadatas = []
    ids = []

    for resume in data["resumes"]: #step through resumes
            resume_id = resume["resume_id"]

            for bullet in resume.get("bullets",[]): #load each bullet if it's labelled
                    text = bullet.get("text")
                    if not text:
                            continue
                    
                    company = bullet.get("company")
                    title = bullet.get("title")

                    role = role_lookup.get((company, title), {}) #set role via lookup table

                    documents.append(text) #append actual bullet text

                    skills = bullet.get("skills",[]) #convert skills dict to comma joined list
                    skills = ", ".join(skills)

                    metadatas.append({ #append metadata
                            "candidate_name": candidate["name"],
                            "resume_id": resume_id,
                            "company": company,
                            "title": title,
                            "dates": role.get("dates",""),
                            "skills": skills,
                            "confidence":bullet.get("confidence","neutral"),
                            "focus": resume.get("focus","")
                    })

                    ids.append(bullet["id"])

    collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
    )

load_collection()