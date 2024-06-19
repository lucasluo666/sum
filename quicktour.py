from typing import List
import torch
from coop import VAE, util

# 模型名称，可以选择 "megagonlabs/bimeanvae-amzn", "megagonlabs/optimus-yelp", "megagonlabs/optimus-amzn"
model_name: str = "megagonlabs/bimeanvae-amzn"
vae = VAE(model_name)

# 输入评论列表
reviews: List[str] = [
    "I love this ramen shop!! Highly recommended!!",
    "Here is one of my favorite ramen places! You must try!"
]

# 编码输入评论为潜在向量
z_raw: torch.Tensor = vae.encode(reviews)

# 输入评论的所有组合
idxes: List[List[int]] = util.powerset(len(reviews))
# 计算所有组合的潜在向量平均值
zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes])

# 生成摘要
outputs: List[str] = vae.generate(zs)
print("Generated Outputs: ", outputs)

# 输入-输出重叠通过ROUGE-1 F1得分衡量
best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
print("Best Summary: ", best)
