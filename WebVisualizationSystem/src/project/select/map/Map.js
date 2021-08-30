import React, { Component, createRef } from 'react'
import * as echarts from 'echarts'
import 'echarts-gl'
import Store from '@/store'
import "./Map.scss"
import chinaJson from './China'
import _ from 'lodash'

//测试数据
const userID = Array.from({ length: 100 });

class Map extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  mapRef = createRef();
  //初始化地图
  initMap = () => {
    // 获取用户id数组
    const users = userID;
    //获取所需绘制的用户柱状图数据，目前为random,之后根据返回的id在数据集中筛选。
    const data = [];
    let maxlat = 117.303484;
    let minlat = 109.664816;
    let maxlon = 25.519951;
    let minlon = 20.223273;
    _.forEach(users, function (item) {
      data.push([Math.floor(Math.random() * (maxlat - minlat + 1) + minlat), Math.floor(Math.random() * (maxlon - minlon + 1) + minlon), Math.random() * 100])
    });
    // 防止OD的weight过大，取一个平方根
    let newData = data.map(function (dataItem) {
      return [dataItem[0], dataItem[1], Math.sqrt(dataItem[2])]
    });
    console.log(newData);
    // 注册地图到组件
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
      //visualMap图例
      visualMap: {
        max: 100, //最大值
        realtime: true, //拖拽时是否实时更新
        calculable: true, //拖拽时是否显示手柄
        inRange: { //颜色数组
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        },
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
        //视角布置
        viewControl: {
          alpha: 60,//初始视角
          minAlpha: 30,//最小视角
          maxAlpha: 90,//最大视角
          distance: 70,//视角到主体的初始距离
          minDistance: 5,//鼠标拉近的最小距离
          maxDistance: 80,//鼠标拉近的最大距离
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
        // 灯光设置
        light: {
          //太阳光参数
          main: {
            intensity: 2,//光照强度
            shadow: true,//阴影
            shadowQuality: 'high',//阴影质量
            alpha: 60,// 光照角度
          },
          //全局环境光
          ambient: {
            color: '#fff', //颜色
            intensity: 0.1, //强度
          },
          //纹理
          ambientCubemap: {
            exposure: 1,//曝光值
            diffuseIntensity: 0.75//漫反射强度
          }
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
          }
        },
        //超采样
        temporalSuperSampling: {
          enabled: true
        }
      },
      series: [{
        type: 'bar3D',
        coordinateSystem: 'geo3D',//坐标系
        shading: 'lambert',// 三维图形的着色效果
        data: data,
        barSize: 0.1, //柱的大小
        minHeight: 0.2,//柱高的最小值
        silent: false, //是否不响应鼠标事件，false未响应，反之为不响应
        itemStyle: {//柱条样式
          opacity: 0.8
        },
        //柱条，鼠标悬浮/点击的标签事件
        // label: {
        //   show: false,
        //   distance: 1,
        //   //显示weight数据
        //   formatter: params=> params.value[2],
        //   //字体格式
        //   textStyle: {
        //     // color: '#000',
        //     fontFamily: 'Microsoft Yahei',
        //     fontSize: 10,
        //     fontWeight: 'normal'
        //   }
        // },
        // 柱条高亮标签样式
        emphasis: {
          label: {
            show: false,//不展示时下方代码无效
            distance: 2,
            formatter: params => params.value[2],
            textStyle: {
              color: '#000',
              fontFamily: 'Microsoft Yahei',
              fontSize: 10,
              fontWeight: 'bolder'
            }
          }
        }
      }]
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

Map.contextType = Store;

export default Map;