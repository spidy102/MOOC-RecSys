import pandas as pd, numpy as np
import torch
import json
from torch import nn
from transformers import BertTokenizer
from transformers import BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

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


cos_mat = np.loadtxt(open("sim.csv"),delimiter=",")

ans = []
for i in range (0,len(cos_mat)):
    for j in range (0, i):
        cos_mat[i][j] = cos_mat[j][i]

df = pd.read_csv("questions.csv", index_col=[0])

questions = df[['Answer(1/0)','Text','Urgency(1-7)','Answer_text','course_display_name']].values
unique_courses = df['course_display_name'].unique()

def grouping(course_name):
    adj_list = dict()
    for i in range (0, len(cos_mat)):
        cur = []
        if (questions[i][4]!=course_name):
            continue
        for j in range (0, len(cos_mat[0])):
            if (questions[j][4]==course_name and cos_mat[i][j]>0.45):
                cur.append(j)
        adj_list[i] = cur
    
    visited = [False]*len(cos_mat)

    def dfs(i):
        visited[i] = True
        neighbors = adj_list[i]
        for j in range (0, len(neighbors)):
            if (visited[neighbors[j]]==False):
                cur_component.append(neighbors[j])
                dfs(neighbors[j])

    cur_component = []
    components = []
    for i in range (0, len(cos_mat)):
        if (questions[i][4]!=course_name):
            continue
        if (visited[i]==False):
            cur_component.append(i)
            dfs(i)
            components.append(cur_component)
            cur_component = []
    
    return components


all_groups = {}

for i in unique_courses:
    components = grouping(i)
    all_groups[i] = components

question_dict = {}

for i in range (0, len(questions)): 
    question_dict[questions[i][1]] = i

def predict_instructor():
    quest_set_glo = {}
    for index in unique_courses:
        cur_course_name = index
        cur_group = all_groups[cur_course_name]
        quest_set = []
        for i in range (0,len(cur_group)): 
            similar_questions_group = cur_group[i]
            question_from_set = ""
            max_urgency = 0
            if (len(similar_questions_group)>1):
                for j in range (0,len(similar_questions_group)):
                    cur_question = similar_questions_group[j]
                    if (questions[cur_question][0]==0 and questions[cur_question][2]>max_urgency):
                        max_urgency = max(max_urgency, questions[cur_question][2])
                        question_from_set = [questions[cur_question][1],question_dict[questions[cur_question][1]]]
                if (question_from_set!=""):
                    quest_set.append(question_from_set)
        quest_set_glo[cur_course_name] = quest_set
    # print(quest_set_glo)
    return quest_set_glo

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def predict_student(question):
    
    new_questions = df[['Text','Answer(1/0)','Answer_text']]
    questions_text = new_questions['Text'].values
    questions_text = np.append(questions_text, question)
    print(questions_text[len(questions_text)-1])
    similar_questions = []
    cosine_similarities = []
    X = vectorizer.fit_transform(questions_text.astype("str"))
    for i in range (0, len(questions_text)-1):
        simi_index = cosine_similarity(X[i],X[len(questions_text)-1])
        cosine_similarities.append(simi_index)
        if (simi_index>0.3):
            print(i)
            if (new_questions.loc[i]['Answer(1/0)']==1):
                similar_questions.append([new_questions.loc[i][0], new_questions.loc[i][2]])
    if (len(similar_questions)==0):
        cosine_similarities.append(1)
        # global cos_mat
        # cos_mat = cos_mat.tolist()
        # cos_mat.append(cosine_similarities)
        # print (cos_mat[0])
        # for i in range (0,len(cos_mat[0])):
        #     cos_mat[i].append(cosine_similarities[i])
        # cos_mat = np.array(cos_mat)
        # global components
        # components = grouping()
        # # have to modify entry in original dataset as well
        # np.savetxt("temp.csv", cos_mat, delimiter=',')
        len(cosine_similarities)
        return [0,cosine_similarities]
    else:
        return [1,similar_questions, cosine_similarities]
    # return [1, [[questions[0][1],questions[0][3]]]]

def ask_question(cosine_similarities, quest, model):
    global cos_mat
    cos_mat = cos_mat.tolist()
    cos_mat.append(cosine_similarities)
    len(cosine_similarities) 
    len(cos_mat[0])
    for i in range (0,len(cos_mat[0])):
        print(i)
        cos_mat[i].append(cosine_similarities[i])
    cos_mat = np.array(cos_mat)
    global components
    components = grouping()
    # have to modify entry in original dataset as well
        
    out = tokenizer(quest, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    att_mask = out['attention_mask']
    input_ids = out['input_ids']
    token_typ_ids = out['token_type_ids']
    output = model(input_ids, att_mask)
    output = output.detach().numpy()
    label  = np.argmax(output[0])
    label = 1+label/2
    new_entry = {'Text':quest, 'Question(1/0)':1, 'Urgency(1-7)':label, 'Answer(1/0)':0, 'Answer_text':""}
    print(new_entry)
    df.loc[len(df)] = new_entry
    df.to_csv("questions.csv")
    np.savetxt("temp.csv", cos_mat, delimiter=',',fmt="%s")
    
def answer_question(id, answer):
    # df.loc[id] = {'Answer_text':answer,'Answer(1/0)':1}
    global components
    print(answer)
    df.loc[id, 'Answer_text'] = answer
    df.loc[id, 'Answer(1/0)'] = 1
    # for i in range (0, len(components)):
    #     if id in components[i]:
    #         cur = components[i]
    #         for i in range (0, len(cur)):
    #             df.loc[cur[i], 'Answer_text'] = answer
    #             df.loc[cur[i], 'Answer(1/0)'] = 1
    df.to_csv("questions.csv")
    # print(df.loc(id))
    
    components = grouping()
    