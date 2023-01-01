from fastapi import FastAPI, Request
from similar_questions_generation import *
from model import *
from fastapi.middleware.cors import CORSMiddleware

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 13)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

# model = BertClassifier()
# torch.load("entire_model.pt")

cos_similarities_arr = []

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/pred_inst")
def pred_inst():
    return predict_instructor()

@app.get("/pred_student")
def pred_student(quest):

    res = predict_student(quest)
    # model.eval()
    global cos_similarities_arr
    if res[0]==0:
        
        cos_similarities_arr = res[1]
        return [0,"Ask question"]
    else: 
        cosine_similarities_arr  = res[2]
        return [1,"We found some similar questions in the database", res[1]] 

@app.post("/answer_quest/{id}")

async def answer_quest(info: Request):
    obj = await info.json()
    answer_question(obj["id"], obj["answer"]) 


@app.post("/ask_quest/")

async def ask_quest(info:Request):
    obj = await info.json()
    ask_question(cos_similarities_arr, obj["quest"], model)