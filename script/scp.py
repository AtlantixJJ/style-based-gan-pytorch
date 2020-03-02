list=[
"record/l1/celebahq_stylegan_linear_l10.0/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l10.0001/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l10.0002/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l10.0004/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l10.0006/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l10.0008/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l10.001/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l11e-05/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l11e-06/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l11e-07/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l13e-05/stylegan_linear_extractor.model",
"record/l1/celebahq_stylegan_linear_l16e-05/stylegan_linear_extractor.model",
]
import os
os.system("mkdir record/l1")
for p in list:
    ind = p.rfind("/")
    dir = p[:ind]
    os.system(f"mkdir {dir}")
    os.system(f"scp img:srgan/{p} {dir}")