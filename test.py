import transformers

tokenizer = transformers.LlamaTokenizer.from_pretrained("/home/zhuminjun01/output1/llama-7B")
model = transformers.LlamaForCausalLM.from_pretrained("/home/zhuminjun01/output1/llama-7B")
model.to("cuda")

batch = tokenizer(
    "Write an alticle about solar system ",
    return_tensors="pt", 
    add_special_tokens=False
)
batch = {k: v.cuda() for k, v in batch.items()}
generated = model.generate(batch["input_ids"], max_length=500)

print(tokenizer.decode(generated[0]))
