import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
app = FastAPI()



@app.get('/')
async def root():
    return {"message": "my name is", "name": "loll"}

@app.post("/properties")
async def get_properties(molecules: bytes = File(), property: str = ""):
    with open("temp_in.txt", "w") as f:
        f.write(molecules.decode())
    os.system(f"python properties.py --prop {property} < temp_in.txt > temp_out.txt")
    return FileResponse("temp_out.txt")

@app.post("/rationales")
async def generate_rationales(actives: bytes = File(), property: str = ""):
    with open("temp_in.txt", "w") as f:
        f.write(actives.decode())
    os.system(f"python mcts.py --prop {property} --ncpu 2 --data temp_in.txt > temp_out.txt")
    return FileResponse("temp_out.txt")

@app.post("/merge_rationales")
async def merge_rationales(rationales_1: bytes = File(), rationales_2: bytes = File()):
    with open("temp_in_1.txt", "w") as f:
        f.write(rationales_1.decode())
    with open("temp_in_2.txt", "w") as f:
        f.write(rationales_2.decode())
    os.system(f"python merge_rationale.py --rationale1 temp_in_1.txt --rationale2 temp_in_2.txt > temp_out.txt")
    return FileResponse("temp_out.txt")

# @app.post("/generate_molecules")
# async def generate_molecules(raionales: bytes = File()):
#     with open("temp_in.txt", "w") as f:
#         f.write(raionales.decode())
#     os.system(f"python decode.py --rationale temp_in.txt --model ckpt/chembl-h400beta0.3/model.20 --num_decode 1000 > temp_out.txt")
#     return FileResponse("temp_out.txt")

@app.post("/generate_modelcules_from_rationales")
async def generate_modelcules_from_rationales(raionales: bytes = File()):
    return FileResponse("output.txt")


