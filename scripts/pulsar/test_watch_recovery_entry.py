import ast,unittest
from pathlib import Path
class WatchRecoveryEntry(unittest.TestCase):
 def test_daily_recovery_uses_same_native_wrapper_and_original_budget(self):
  tree=ast.parse((Path(__file__).parent/'watch_pipeline.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='execute']
  commands=[(ast.literal_eval(n.args[0]),ast.literal_eval(n.args[1])) for n in calls if isinstance(n.args[0],ast.List) and all(isinstance(x,ast.Constant) for x in n.args[0].elts)]
  self.assertIn((['/bin/bash','scripts/pulsar/daily_update.sh'],1200),commands)
 def test_native_wrapper_retains_model_and_publication_contract(self):
  text=(Path(__file__).parent/'daily_update.sh').read_text();self.assertIn('export TRADING_DEEPSEEK_REPAIR_MODEL=deepseek-v4-pro',text);self.assertIn('--job daily --publish --allow-verified-partial',text)
