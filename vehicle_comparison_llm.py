from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class VehicleComparisonLLM:
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    system_prompt = """
        You are an expert automotive analyst. Your task is to process a list of car comparison batches. 

        For each batch containing 5 car listings:
        1. Compare the 5 offered cars based on all provided technical and listing details (price, year, mileage, engine power/capacity, condition/extra_info, etc.).
        2. Write a comprehensive, well-structured comparison analysis. The response for each batch MUST be a substantive paragraph containing between 4 and 6 sentences. Do not make it too brief, but keep it concise and analytical.
        3. Highlight key trade-offs (e.g., higher mileage vs. lower price, newer production year vs. engine power, condition history).
        4. Strictly return your output in a valid JSON array format, where each element corresponds to a batch ID and contains the comparison text.
        
        Output format requirement:
        [
          {
            "batch_id": 1,
            "comparison": "Detailed 4-6 sentence analysis comparing car 1 to car 5..."
          }
        ]
    """

    user_prompt = """
        Compare the following 5 car offers and provide a detailed analysis:\n
    """

    def __init__(self, additional_weights_path: str):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = PeftModel.from_pretrained(base_model, additional_weights_path)
        self.model.eval()


    def __call__(self, retrieved_records: list) -> str:
        formatted_prompt = self._format_to_chat(retrieved_records)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


    def _format_to_chat(self, user_input: list):
        messages = [
            {"role": "user", "content": self.user_prompt + ''.join(user_input)},
            {"role": "assistant", "content": self.system_prompt}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False)