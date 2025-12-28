import json
from typing import List
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from chromadb import HttpClient

import os
from dotenv import load_dotenv

from collections import defaultdict

import traceback

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

#get chromadb settings from env
chroma_host = os.getenv("CHROMA_HOST")
chroma_port_str = os.getenv("CHROMA_PORT")
chroma_port = int(chroma_port_str)
chroma = chromadb.HttpClient(host=chroma_host,port=chroma_port)

collection = chroma.get_or_create_collection(
    name="resume_bullets",
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
    return results["documents"][0]

#make the resume
def generate_bullets(job_requirements: dict, bullets: List[str]):
    #complete prompt using json job reqs and resume bullets from collection
    prompt = RESUME_GENERATION_PROMPT.format( 
          job_requirements=json.dumps(job_requirements, indent=2),
          bullets="\n".join(f"- {b}" for b in bullets)
    )

    response = client.chat.completions.create(
          model="gpt-4.1",
          messages=[{"role":"user", "content": prompt}],
          response_format={"type":"json_object"}
    )

    return json.loads(response.choices[0].message.content)

#align the new bullets to the job roles for for final resume
def match_bullets_to_roles(aligned_bullets, threshold=0.85):
    matched = []

    for text in aligned_bullets["rewritten_bullets"]: #grab the original bullet that best aligns with new one
        print(text)
        result = collection.query(
            query_texts=[text],
            n_results=1
        )

        #grab similarity score and info from original match
        score = result["distances"][0][0]
        original_id = result["ids"][0][0]
        metadata = result["metadatas"][0][0]

        print(f"Original ID = {original_id}")

        #add the metadata back to the bullets
        matched.append({
            "rewritten_text": text,
            "original_bullet_id": original_id,
            "title": metadata["title"],
            "company": metadata["company"],
            "dates": metadata["dates"]
        })

    #sort bullets into roles
    roles = {} #created roles

    for entry in matched:
        title = entry['title']

        #create role and fields if it hasn't been seen yet
        if title not in roles:
            roles[title] = {
                "company": entry['company'],
                "title": title,
                "dates": entry['dates'],
                "experiences": []
            }

        roles[title]["experiences"].append(entry['rewritten_text'])
        
    
    print(json.dumps(roles, indent=2))
    return roles

#check for missing keywords and skills
"""
def critic_pass(generated_resume:str):
    response = client.chat.completions.create(
         model="gpt-4.1-mini",
         messages = [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": generated_resume}
         ],
         response_format={"type":"json_object"}
    )

    return json.loads(response.choices[0].message.content)
"""
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

    #get bullets from collection that matches skills from jd
    bullets = retrieve_relevant_bullets(job_req["required_skills"])

    #generate resume and critique
    aligned_bullets = generate_bullets(job_req, bullets)
    #critique = critic_pass(resume)

    experience = match_bullets_to_roles(aligned_bullets)
    #print(json.dumps(experience,indent=2))
    #print("\n Aligned Bullets \n")
    #print(aligned_bullets)

    #
    #print("\n Critic Report  \n")
    #print(json.dumps(critique,indent=2))