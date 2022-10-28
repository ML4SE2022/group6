(make py-train-token || echo "Failed to train py token on py"); \
(make py-train-token-on-js || echo "Failed to train py token on JS"); \
(rm save/py150/train_blocksize_1024_wordsize_1_rank_0); \
(make py-train-token-on-ts || echo "Failed to train py token on TS"); \
(rm save/py150/train_blocksize_1024_wordsize_1_rank_0); \
(make ts-train-token || echo "Failed to train java token on TS"); \
(make js-train-token || echo "Failed to train java token on JS"); \
echo "Done!"
