import utils
n = 16
id2cid = utils.create_id2cid(n, list(range(1, n)), [1] * n)
print(id2cid)
n = 19
id2cid = utils.create_id2cid(n, [5, 7, 9], [4, 6, 8])
cid2id = utils.create_cid2id(n, [5, 7, 9], [4, 6, 8])
print(id2cid)
print(cid2id)