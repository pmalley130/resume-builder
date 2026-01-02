This is a simple python and OpenAI-powered resume builder. It ingests candidate data and a job description to generate an ATS-aligned resume (including experience, skills, and a professional summary).

To use, run resume_builder.py to save a JSON version of your new resume to data/new_resume.json. Run render_resume.py to generate an HTML resume from that JSON to output/resume.html

Requirements:
* OpenAI AI Key
* Chromadb container
* Job description text file saved to data/job_description.txt
* JSON describing the candidate saved to data/resume_data.json (see sample file for schema)

To-do/Wishlist:
* Have OpenAI parse a description from a web page/URL instead of a txt file
* Better data formatting so refactoring makes more sense (lots of copy and paste has to be done in the resume_data.json file)
* On that note, workflow to parse resumes into resume_data in the first place
* Better debugging messaging and error capturing
* PDF render
* Interactive script

For more information, [see my blog](https://learnwithpatrick.casa/building-an-ai-resume-builder/)