# API调用
后端只需要关注api.py文件即可
1. rec_init初始化获得cf对象
2. 调用cf对象的recommend
3. recommend函数参数（top:输出的项目总数，需要大于rating_dict非零评分的项目数；rating_dict:针对某用户的评分字典；shuffle:推荐内容是否打乱顺序，不打乱的话同类型的推荐会连在一起）