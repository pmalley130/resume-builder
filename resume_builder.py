import json
from typing import List
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from chromadb import HttpClient

import os
from dotenv import load_dotenv

import traceback

from prompts import ( #import prompts from separate file
    JD_EXTRACTION_PROMPT,
    RESUME_GENERATION_PROMPT
)

def embed_texts(texts: list[str]) -> list[list[float]]:
      response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
      )
      return [e.embedding for e in response.data]

#load collection
def load_collection(collection, path="data/resume_data.json"):

    #load candidate data from file
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

    #add to collection
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    except Exception as e:
          print(f"Error occured: {e}")
          traceback.print_exc

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

    print("Job Description Parsed")
    return json.loads(response.choices[0].message.content)

#grab my skills from the chroma collection
def retrieve_relevant_bullets(skills: List[str], k=20):
    query = " ".join(skills)
    results = collection.query(
       query_texts=[query],
       n_results=k   
    )
    print("Relevant Bullets retrieved")
    return results["documents"][0]

#make the resume
def generate_bullets_and_skills(job_requirements: dict, bullets: List[str]):
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

    answer = json.loads(response.choices[0].message.content)
    print("New Bullets, Skills, and Summary Generated")
    return answer['rewritten_bullets'], answer['targeted_skills'], answer['professional_summary']

#align the new bullets to the job roles for for final resume
def match_bullets_to_roles(aligned_bullets):
    matched = []

    for text in aligned_bullets: #grab the original bullet that best aligns with new one
        result = collection.query(
            query_texts=[text],
            n_results=1
        )

        #grab info from original match
        original_id = result["ids"][0][0]
        metadata = result["metadatas"][0][0]

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
        
    
    #print(json.dumps(roles, indent=2))
    return roles

#load name, portfolio, and education
def load_static_data(path="data/resume_data.json"):
    
    with open(path,encoding='utf8') as f:
        data = json.load(f)

    candidate = {}
    candidate['name'] = data['candidate']['name']
    candidate['location'] = data['candidate']['base_location']
    candidate['education'] = data['candidate']['education']
    candidate['portfolio'] = data['candidate']['portfolio_links']
    candidate['certifications'] = data['candidate']['certifications']

    return candidate

def load_experiences():
    #check to see if we've done this part
    if os.path.exists("data/aligned_experiences.json"):
        with open("data/aligned_experiences.json", 'r') as f:
            saved_data = json.load(f)
        print("Previous experiences loaded")
        return saved_data['experience'],saved_data['targeted_skills'],saved_data['professional_summary']
    else:
        #populate collection with bullets
        print("Loading collection")
        load_collection(collection)
        print("Collection loaded")
        #open job description and parse into relevant json based on prompt
        with open("data/job_description.txt") as f:
            jd_text = f.read()
        job_req = parse_jd(jd_text)

        #list to hold generated experience and skills
        #get old bullets from collection that matches skills from jd
        bullets = retrieve_relevant_bullets(job_req["required_skills"])

        #generate bullets, skills, and summary for new resume
        aligned_bullets, skills, summary = generate_bullets_and_skills(job_req, bullets)
        
        #create experience section for resume
        experience = match_bullets_to_roles(aligned_bullets)

        save_data = {}
        save_data['professional_summary'] = summary
        save_data['experience'] = experience
        save_data['targeted_skills'] = skills
        
        #write to file for later
        with open("data/aligned_experiences.json", 'w') as f:
             json.dump(save_data, f, indent=4)

        return experience, skills, summary
    
#index bullets to roles for padding
def index_resume_data(path="data/resume_data.json"):
    role_index = {}
    seen = {}

    with open(path,encoding='utf8') as f:
        data = json.load(f)

    print("Indexing Role Data")
    #create one role slot per role in data
    for role in data.get("candidate", {}).get("roles",[]):
        title = role.get("title")
        role_index[title] = []
        seen[title] = set()

    #add bullets by walking through, checking for dupes, and assigning them to matching role
    for resume in data.get("resumes"):
        for bullet in resume.get("bullets"):
            title = bullet.get("title")
            text = bullet.get("text")
            skills = bullet.get("skills") or []

            if text in seen[title]:
                continue

            role_index[title].append({
                 "text": text,
                 "skills": [s for s in skills]
            })
            seen[title].add(text)

    return role_index

#ensure that generated resume shows more than one role and enough bullets
def pad_roles(
          experience, role_index, min_roles=3, min_bullets=4, path="data/resume_data.json"
):
    #read original data to backfill info
    with open(path,encoding='utf8') as f:
        resume_data = json.load(f)
    
    #sort reverse chron so we always get the latest jobs
    roles_sorted = sorted(
         resume_data["candidate"]["roles"],
         key= lambda r: r.get ("start", ""),
         reverse=True
    )
    #if we don't have enough roles, walk through role index and add them if they're not already represented until we have enough
    if len(experience) < min_roles:
        for role in roles_sorted:
            title = role["title"]
            if title not in experience:
                 experience[title] = {
                      "company": role.get("company"),
                      "title": title,
                      "dates": role.get("dates"),
                      "experiences": []
                 }
            if len(experience) >= min_roles:
                break

    #now add bullets for each role until we have enough, skipping ones with overlapping skills
    for role_title, role_block in experience.items():
        role_block.setdefault("experiences",[])

        bullets = role_block["experiences"]
        used_text= set(bullets)
        used_skills = set()

        #add fillers matching the role, skipping previously used fillers and overlapped skills
        for candidate in role_index.get(role_title, []):
            if len(bullets) >= min_bullets:
                break

            text = candidate["text"]
            skills = set(candidate.get("skills", []))

            if text in used_text:
                continue

            if skills and (skills & used_skills):
                continue

            bullets.append(text)
            used_text.add(text)
            used_skills |= skills

    return experience

#main flow
if __name__ == "__main__":
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
    resume = {}


    #load things we don't need ai for (on every resume)
    resume = load_static_data()

    #index role data
    role_index = index_resume_data()

    #have AI generate experience, skills, and summary
    experience, skills, summary = (load_experiences())
    
    #pad roles and bullets
    experience = pad_roles(experience, role_index)

    #add experiences and skills to resume
    resume['experiences'] = experience
    resume['skills'] = skills
    resume['professional_summary'] = summary

    #save new resume to json
    with open("data/new_resume.json", 'w') as f:
        json.dump(resume, f, indent=4)

    print("New Resume as JSON \n")
    print(json.dumps(resume,indent=3))


    
       