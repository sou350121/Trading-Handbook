import contextlib,datetime as dt,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import local_index as m
class CompactIndex(unittest.TestCase):
 def setup_case(self,folder):
  root=Path(folder);path=root/'index.json';manifest=root/'manifest.json';day=dt.datetime.now(dt.timezone.utc).date().isoformat();text='Operator liquidation risk and portfolio simulation evidence.'
  obj,_=m.make_index([dict(id='compact-fixture',source='guide.md',text=text,date=day,source_type='trading_handbook',section='Evidence')])
  def git(*a):
   if a[0]=='ls-tree':return 'guide.md\n'
   if a[0]=='show' and a[1]=='-s':return day+'T00:00:00+00:00'
   if a[0]=='show':return text
   raise AssertionError(a)
  patches=patch.multiple(m,ROOT=root,PATH=path,MANIFEST=manifest,published_head=lambda expected=None:'a'*40,git=git,make_index=lambda chunks:(obj,0))
  return path,manifest,obj,patches
 def test_real_bm25_roundtrip_fits_existing_bound_without_losing_search(self):
  with tempfile.TemporaryDirectory() as tmp:
   path,manifest,obj,env=self.setup_case(tmp);compact=json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode();pretty=json.dumps(obj,ensure_ascii=False,indent=2).encode();self.assertGreater(len(pretty),len(compact))
   with env,patch.object(m,'MAX_INDEX_BYTES',(len(compact)+len(pretty))//2):
    m.build();self.assertEqual(json.loads(path.read_bytes()),obj);loaded=m.check();hits=m.search_index(loaded,'operator liquidation',days=365,top=5);self.assertTrue(hits);self.assertEqual(hits[0]['source'],'guide.md');self.assertEqual(json.loads(manifest.read_bytes())['index_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
 def test_genuinely_oversized_compact_data_keeps_previous_files(self):
  with tempfile.TemporaryDirectory() as tmp:
   path,manifest,obj,env=self.setup_case(tmp);path.write_bytes(b'previous-index');manifest.write_bytes(b'previous-manifest')
   with env,patch.object(m,'MAX_INDEX_BYTES',1):
    with self.assertRaisesRegex(ValueError,'index_too_large'):m.build()
   self.assertEqual(path.read_bytes(),b'previous-index');self.assertEqual(manifest.read_bytes(),b'previous-manifest')
