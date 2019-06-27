docker run \
--mount type=bind,source=$(pwd)/src,target=/InstagramWall/src \
--mount type=bind,source=$(pwd)/lib,target=/InstagramWall/lib \
--mount type=bind,source=$(pwd)/data,target=/InstagramWall/data \
-it wall
