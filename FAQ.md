
---

peft:


```
module 'torch_npu.npu' has no attribute 'mem_get_info'
```

解决办法：
```
pip3 install torch-npu==2.1.0.post3
```




---

deepspeed:


```
/workspace/installs/conda/envs/llm-dev/lib/python3.9/site-packages/torch_npu/include/torch_npu/csrc/core/npu/interface/AclInterface.h:159:30: note: suggested alternative: ‘aclopAttr’
 aclError AclGetCannAttribute(aclCannAttr cannAttr, int32_t *value);
                              ^~~~~~~~~~~
                              aclopAttr
/workspace/installs/conda/envs/llm-dev/lib/python3.9/site-packages/torch_npu/include/torch_npu/csrc/core/npu/interface/AclInterface.h:159:60: error: expected primary-expression before ‘*’ token
 aclError AclGetCannAttribute(aclCannAttr cannAttr, int32_t *value);
                                                            ^
/workspace/installs/conda/envs/llm-dev/lib/python3.9/site-packages/torch_npu/include/torch_npu/csrc/core/npu/interface/AclInterface.h:159:61: error: ‘value’ was not declared in this scope
 aclError AclGetCannAttribute(aclCannAttr cannAttr, int32_t *value);
                                                             ^~~~~
```


```
pip3 install torch-npu==2.1.0
```


----



```
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

https://github.com/huggingface/accelerate/issues/2368
```

华为好像是不支持单进程多卡。






