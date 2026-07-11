from pathlib import Path
import ast, re, yaml, json
root=Path.cwd()
backbone=(root/'src/model/backbone.py').read_text()
agg=(root/'src/vlm/aggregate.py').read_text()
cache=(root/'src/model/cache_features.py').read_text()
corn=yaml.safe_load((root/'configs/corn.yaml').read_text())
assert 'dinov3_vits16' in backbone
assert 'dinov2_vits14_reg' in backbone
assert 'POINT_DECISION_THRESHOLD = 0.39' in agg
assert 'feature_dim' in cache and '384' in cache
assert corn['backbone']['name'] in {'dinov2_vits14_reg','dinov2_vits14','dinov3_vits16'}
print(json.dumps({
  'backbone_support': ['dinov2_vits14_reg','dinov2_vits14','dinov3_vits16'],
  'configured_backbone': corn['backbone']['name'],
  'point_decision_threshold': 0.39,
  'cache_feature_dim_contract': 384,
}, indent=2))
