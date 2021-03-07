# SharedCodedMatching
Point Cloud Registration for matching marked points.

算法适用于两片有同名点的点云之间的匹配，其中必须包含一个以上的已知同名点。
算法原理是基于pnp实现二维与三维点之间的重投影误差估计。

需要准备的第三方库：
-PCL
-openCV

另外需要的算法：
-标志点识别算法
