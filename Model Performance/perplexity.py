from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from torch.nn import functional as F
from tqdm import tqdm

class Perplexity:
    
    def __init__(self,model,data):
        """
        init data and model with AutoTokenizer by transformers model
        """
        
        self.data = data
        
        # self.config = AutoConfig.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def cal_perplexity(self, stride):
        """
        Calculate model's perplexity
        """

        self.encodings = self.tokenizer((self.data), return_tensors="pt")

        self.max_length = self.model.config.n_positions
        self.stride = stride
        self.seq_len = self.encodings.input_ids.size(1)

        nlls = []
        self.prev_end_loc = 0

        for begin_loc in tqdm(range(0, self.seq_len, self.stride)):
            end_loc = min(begin_loc + self.max_length, self.seq_len)
            trg_len = end_loc - self.prev_end_loc  # may be different from stride on last loop
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels= target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == self.seq_len:
                break

        self.ppl = torch.exp(torch.stack(nlls).mean())

        return self.ppl

    # def cal_perplexity(self):
    #     """
    #     Calculate model's perplexity
    #     """

    #     self.inputs = self.tokenizer(self.data, return_tensors = "pt")
    #     self.loss = self.model(input_ids = self.inputs["input_ids"], labels = self.inputs["input_ids"]).loss
    #     self.ppl = torch.exp(self.loss)

    #     return self.ppl

if __name__ == "__main__":

    stride = 512

    model = 'gpt2'
    text = 'ยิ่งโตขึ้นเรายิ่งใช้บทเพลง "ฉันมายินดีให้กับรักที่สดใส~" บ่อยขึ้นตามไปด้วย เพราะคนดังที่เราชอบกำลังทยอยเป็นฝั่งเป็นฝาไปตาม ๆ กัน \
            ซึ่งตอนนี้ก็ถึงคิวของพ่อหนุ่ม เพอร์ซีย์ แจ็กสัน ของเราอย่าง Logan Lerman แล้วค่ะ \
            เมื่อล่าสุด Ana Corrigan แฟนสาวของเขาได้ออกมาประกาศผ่านอินสตาแกรมว่าทั้งคู่หมั้นหมายกันเป็นที่เรียบร้อยแล้ว พร้อมแคปชั่นว่า \
            "Thats Mrs Logie to you" \
            Ana ไม่ได้เป็นดาราคนดังในวงการ แต่เธอมีชื่อเสียงในด้านการออกแบบศิลปะโคมไฟเซรามิก เธอกับ Logan เปิดตัวเดตกันครั้งแรกในปี 2020 และคบกันแบบเงียบ ๆ ไม่ได้หวือหวาอะไรมาก \
            จนมาถึงตอนนี้ทั้งสองตัดสินใจแต่งงานกันเป็นที่เรียบร้อยแล้ว ขอแสดงความยินดีด้วยน้าา' 
    
    Calculate_Test = Perplexity(model,text)
    perplexity = Calculate_Test.cal_perplexity(stride= stride)
    
    print(f"Perplexity of {model} model is {perplexity}")