import React, { Component ,createRef} from 'react'
import * as echarts from 'echarts'
import 'echarts-gl'
import "./Map.scss"
import chinaJson from './China'

class Map extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  mapRef = createRef();
  //初始化地图
  initMap = () => {
    echarts.registerMap('China', chinaJson);
    const myMap = echarts.init(this.mapRef.current);
    const option = {
      // 鼠标悬浮的字体样式
      textStyle: {
        color: '#000',
        fontFamily: 'Microsoft Yahei',
        fontSize: 12,
        fontWeight: 'bolder'
      },
      // geo3D地图
      geo3D: {
        map: 'China',
        // 行政区边界
        itemStyle: {
          opacity: 1,
          borderWidth: 1,
          bordercolor: '#333'
        },
        // 三维图形的着色效果
        shading: 'lambert',
        //
        viewControl: {
          alpha: 60,//初始视角
          minAlpha: 30,//最小视角
          maxAlpha: 90,//最大视角
          distance: 70,//视角到主体的初始距离
          minDistance: 30,//鼠标拉近的最小距离
          maxDistance: 100,//鼠标拉近的最大距离
          //自动旋转动画
          // autoRotate: true,
          // antoRotateDirection: 'ccw',//逆时针
          // autoRotateSpeed: 2, //旋转速度，即一秒转2度
          // autoRotateAfterStill: 5, //鼠标交互事件5s后继续开始动画
          panMouseButton: 'left',//左键移动地图
          rotateMouseButton: 'right',//右键旋转地图
          rotateSensitivity: 1,//旋转操作的灵敏度
          panSensitivity: 0.7,//平移操作的灵敏度
          zoomSensitivity: 1//缩放操作的灵敏度
        },
        //特效处理
        postEffect: {
          enable: true,
          //高光
          bloom: {
            enable: false
          },
          //环境光
          SSAO: {
            radius: 1,
            intensity: 1,
            enable: true
          },
          //景深
          // depthOfField: {
          //   enable: true,
          //   focalDistance: 100,
          //   focalRange: 200,
          //   blurRadius: 10,
          //   fstop: 5
          // }
        },
        //超采样
        temporalSuperSampling: {
          enabled: true
        }
        //显示地面颜色
        // groundPlane: {
        //   show: true,
        //   // color: '#000'
        // }
      },
    };
    myMap.setOption(option);
  };


  componentDidMount() {
    this.initMap();
  }

  render() {
    return (
        <div
            className="select-map-ctn"
            ref={this.mapRef}
            style={{
              height: '100%',
              width: '100%'
            }}
        ></div>
    );
  }
}

export default Map;