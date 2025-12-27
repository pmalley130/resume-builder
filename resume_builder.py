import json
from typing import List
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

import os
from dotenv import load_dotenv

import traceback

from prompts import ( #import prompts from separate file
    JD_EXTRACTION_PROMPT,
    RESUME_GENERATION_PROMPT,
    CRITIC_PROMPT
)


load_dotenv() #load API key
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"

client = OpenAI() #start openAI client

#vector store setup
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

chroma = chromadb.PersistentClient(path="./chroma_db")

collection = chroma.get_or_create_collection(
    name="resume_bullets",
    embedding_function=embedding_fn
)

collection = chroma.get_or_create_collection(
    name="fresh_resume_test", 
    embedding_function=embedding_fn
)

def embed_texts(texts: list[str]) -> list[list[float]]:
      response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
      )
      return [e.embedding for e in response.data]

#load collection
def load_collection(collection, path="data/resume_data.json"):

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

 
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    except Exception as e:
          print(f"Error occured: {e}")
          traceback.print_exc

    print("collection loaded")

#parse job description into json
def parse_jd(jd_text: str) -> dict:
    response = client.chat.completions.create( #send the job description (as json/dict) and extraction prompt to gpt
            model="gpt-4.1-mini",
            messages=[
                 {"role": "system", "content": JD_EXTRACTION_PROMPT},
                 {"role": "user", "content": jd_text}
            ],
            response_format={"type":"json_object"} #ensure gpt only sends back json, otherwise we get json decode errors
    )

    print(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)

#grab my skills from the chroma collection
def retrieve_relevant_bullets(skills: List[str], k=10):
    query = " ".join(skills)
    results = collection.query(
       query_texts=[query],
       n_results=k   
    )
    print(results["documents"][0])
    return results["documents"][0]

#make the resume
def generate_resume(job_requirements: dict, bullets: List[str]):
    #complete prompt using json job reqs and resume bullets from collection
    prompt = RESUME_GENERATION_PROMPT.format( 
          job_requirements=json.dumps(job_requirements, indent=2),
          bullets="\n".join(f"- {b}" for b in bullets)
    )

    response = client.chat.completions.create(
          model="gpt-4.1",
          messages=[{"role":"user", "content": prompt}]
    )

    return response.choices[0].message.content

#main flow
if __name__ == "__main__":
    #populate collection with bullets
    print("Loading collection")
    load_collection(collection)
    print("collection loaded")
    #open job description and parse into relevant json based on prompt
    with open("data/job_description.txt") as f:
        jd_text = f.read()
    job_req = parse_jd(jd_text)
    print(job_req)
    #get bullets from collection that matches skills from jd
    bullets = retrieve_relevant_bullets(job_req["required_skills"])

    #generate resume
    resume = generate_resume(job_req, bullets)

    print("\n Generated Resume \n")
    print(resume)