#!/usr/bin/env python3
"""Use qwen ultrathink to design the Trading-Handbook taxonomy from all 399 titles."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _qwen import ask

ROOT = pathlib.Path(__file__).resolve().parents[2]
idx = json.loads((ROOT/"data"/"index.json").read_text())
titles = "\n".join(f"{a['pos']:>3} | {a['date']} | {a['title']}" for a in idx)

SYSTEM = (
"你是量化投资 + 机器学习领域的资深架构师，正在为一个名为 Trading-Handbook（量化交易手册）"
"的开源知识库设计内容本体（ontology）与目录结构。这个手册是『照见 / Pulsar』系列的第四册，"
"与 VLA-Handbook、Spatial-Intelligence-Handbook、Physics-Generation-Handbook 共用同一套架构范式：\n"
"- Mintlify 多 tab 导航；\n"
"- 一套『N 轴本体论』，让读者可以把任意两个方法放到同样的坐标轴上做 Pareto 对比（这是手册的灵魂）；\n"
"- Foundations：按『方法族 zone』组织，每个 zone 下面是若干篇方法解构（dissection）；\n"
"- Crossing：跨 zone 的张力图 / 谱系 / 取舍 atlas（例如 A vs B 的本质冲突）；\n"
"- Benchmarks / Deployment / Use-cases / Companies / Frontier。\n"
"读者画像：已经做过实盘或回测的量化研究员 / ML 工程师，不需要科普，要的是『方法间对比、失效模式、"
"如何组合』。\n"
"你的设计必须扎根于下面这 399 篇真实文章语料（这是 QuantML 公众号两年的全部论文导读），"
"让 zone 划分能真正容纳这些文章，覆盖均衡、不重不漏。"
)

PROMPT = f"""下面是语料库的全部 399 个标题（pos | 日期 | 标题）：

{titles}

请深度思考后，设计这个 Trading-Handbook 的完整内容本体，并【只输出一个 JSON 对象】，schema 如下：

{{
  "ontology": {{
    "name": "本体论名字（例如 量化方法五轴）",
    "rationale": "为什么选这几条轴：一句话",
    "axes": [
      {{"id":"axis-slug","name":"轴名","poles":"一端↔另一端","what_it_discriminates":"这条轴在语料里真正区分了什么"}}
    ]
  }},
  "foundations_zones": [
    {{"slug":"kebab-case","name":"中文 zone 名","scope":"这个 zone 收什么方法","est_articles":数字（语料里大致几篇属于它）,"sample_pos":[语料中代表性文章的 pos 数字, 3-6个]}}
  ],
  "crossing_atlases": [
    {{"slug":"kebab-case","name":"张力/谱系名","tension":"它对比的两端本质冲突是什么"}}
  ],
  "benchmarks_categories": [{{"slug":"...","name":"...","scope":"..."}}],
  "deployment_topics": [{{"slug":"...","name":"...","scope":"..."}}],
  "use_cases": [{{"slug":"...","name":"...","scope":"..."}}],
  "frontier_themes": [{{"slug":"...","name":"...","scope":"..."}}],
  "notes": "对人类维护者的 3-5 条关键设计说明"
}}

要求：
- axes 给 4-6 条，必须是语料里真正有方差的维度（信号源 / 预测目标 / 时间尺度 / 模型类 / edge 类型 / 自动化程度 之类，但你自己判断哪几条最有区分力）。
- foundations_zones 给 8-12 个，est_articles 之和应接近 399（允许一篇归一个主 zone）。zone 要按『方法族』而不是按资产类别切。
- 全部 slug 用英文 kebab-case，name 用中文。
- 只输出 JSON，不要解释，不要 markdown 代码围栏。"""

print("calling qwen (ultrathink, may take 1-3 min)...", file=sys.stderr, flush=True)
raw = ask(PROMPT, system=SYSTEM, thinking=True, temperature=0.4, max_tokens=12000, timeout=400)
(ROOT/"data"/"taxonomy.raw.txt").write_text(raw)
# strip fences if any
t = raw.strip()
if t.startswith("```"):
    t = t.split("```",2)[1]
    if t.startswith("json"): t = t[4:]
    t = t.rsplit("```",1)[0] if "```" in t else t
i, j = t.find("{"), t.rfind("}")
obj = json.loads(t[i:j+1])
(ROOT/"data"/"taxonomy.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2))
print("WROTE taxonomy.json", file=sys.stderr)
# summary
print("\n=== ONTOLOGY:", obj["ontology"]["name"], "===")
for ax in obj["ontology"]["axes"]:
    print(f"  • {ax['name']}: {ax['poles']}")
print(f"\n=== FOUNDATIONS ZONES ({len(obj['foundations_zones'])}) ===")
tot=0
for z in obj["foundations_zones"]:
    tot+=z.get("est_articles",0)
    print(f"  [{z['slug']}] {z['name']} (~{z.get('est_articles','?')})")
print(f"  est total = {tot}")
print(f"\n=== CROSSING ({len(obj['crossing_atlases'])}) ===")
for c in obj["crossing_atlases"]:
    print(f"  • {c['name']}")
