# ransac_sandbox
Demonstrate RANSAC for fitting lines and interest point correspondences.

## Line-fitting demo

Run: `> python ransac_line_demo.py` to start the RANSAC line fitter demo.  The demo will:
* Create of a dataset:
  * 100 (noisy) points on a line 
  * 100 random outliers points 

* Plot of each iteration of the RANSAC algorithm as it attempts to find the line:
![ransac_line_demo](/assets/ransac_line_demo.png)


* After 20 iterations, the results are plotted, showing:
    * the best model found (the one resulting in the most inliers),
    * the final model (estimated from that largest set), and
    * the least-squares fit to the data, obviously not fit the line in a way that is robust to the outliers:
![ransac_line_final](/assets/ransac_line_final.png)

The output shows the parameters used to generate the data, and the recovered parameters, which should be close, up to sign:

```
RANSAC found a solution on iteration 14:
        Inliers: 109 of 200 (54.5 %)

Estimated params of line:
        -0.856*x + 0.388*y + 0.342 = 0

True params of line:
        0.860*x + -0.357*y + -0.365 = 0
```

### Animation options:

The following can be set by changing the value of the `animate_pause_sec` parameter at the bottom of the demo script to any of these values:
* None: no plot / animation,
* 0: pause, wait for user interaction between each iteration, or
* (any number > 0): pause this many seconds between each frame.

