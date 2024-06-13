import json

input_dir = "/home/guodong.li/workspace/data/alpaca_gpt4_data_zh.json"
output_dir = "/home/guodong.li/workspace/data/new/train_data_20240105_1w_standard.json"

list_data_dict = json.load(open(input_dir, "r"))

result = []

i = 0
for temp in list_data_dict:

    instruction = temp.get("instruction")
    input_str = temp.get("input")
    output = temp.get("output")
    user = instruction + ( "\n" + input_str if input_str is not None and len(input_str) > 0 else "")

    item = {
        "id": "identity_"+str(i),
        "conversations": [
            {
                "from": "system",
                "value": ""
            }, {
                "from": "user",
                "value": user
            }, {
                "from": "assistant",
                "value": output
            }
        ]
    }
    result.append(item)
    i+=1
    if i==10000:
        break

with open(output_dir, 'w', encoding='utf-8') as b:
    json.dump(result, b, ensure_ascii=False, indent=4)


