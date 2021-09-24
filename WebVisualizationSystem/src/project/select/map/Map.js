import React, {Component, createRef} from 'react'
import * as echarts from 'echarts'
import 'echarts-gl'
import {message} from "antd";
import Store from '@/store'
import {eventEmitter} from '@/common/func/EventEmitter';
import _ from 'lodash';
import "./Map.scss";
//测试数据
import regionJson from './regionJson/Shenzhen';
import userLocations from '../charts/bottom/jsonData/userLoctionCounts'

let myMap = null;

class Map extends Component {
    static contextType = Store;
    constructor(props) {
        super(props);
        this.state = {
            SelectedUsers: [], //用于记录历史的selectedUsers, 用于判断是否重新渲染
        };
    }
    mapRef = createRef();

    // 组织用户-top5位置数据：{id1:[{lnglat:[], count:XX}, {lnglat:[], count:XX},...], id2:[{lnglat:[], count:XX}, {lnglat:[], count:XX},...] ....}
    getUserData = (selectedUsers) => {
        //重新组织数据形式，便于后续筛选
        let data = {};
        _.forEach(userLocations, function (item) {
            data[item.id] = item.data
        });
        //存储筛选的用户数据: [{id:xx, locations: [{lnglat:[], count:xx}, {lnglat:[], count:xx},...]},...]
        let userData = [];
        for(let i = 0; i < selectedUsers.length; i++){
            userData.push({id: selectedUsers[i], locations: data[selectedUsers[i].toString()]})
        }
        return userData;
    };

    //初始化地图
    initMap = () => {
        //绘制基础地图的option参数
        const option = {
            tooltip: {
                formatter: function (params) {// 说明某日出行用户数量
                    return '经度: ' + params.value[0] + '<br />'
                        + '纬度: ' + params.value[1] + '<br />'
                        + '出行次数：' + params.value[2];
                },
            },
            // 鼠标悬浮的字体样式
            textStyle: {
                color: '#000',
                fontFamily: 'Microsoft Yahei',
                fontSize: 12,
                fontWeight: 'bolder'
            },
            //visualMap图例
            visualMap: {
                left: this.props.leftWidth,
                bottom: this.props.bottomHeight,
                max: 500, //最大值
                realtime: true, //拖拽时是否实时更新
                calculable: true, //拖拽时是否显示手柄
                inRange: { //颜色数组
                    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
                },
                textStyle: {
                    color: '#fff',
                    fontWeight: 'normal'
                }
            },
            // geo3D地图
            geo3D: {
                map: 'Shenzhen',
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
                    distance: 90,//视角到主体的初始距离
                    minDistance: 5,//鼠标拉近的最小距离
                    maxDistance: 150,//鼠标拉近的最大距离
                    center: [-10, 0, 18],//视角中心点，初始化地图位置
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
            series:[{
                type: 'bar3D',
                name: '柱子',
                coordinateSystem: 'geo3D',//坐标系
                shading: 'lambert',// 三维图形的着色效果
                data: [],
                barSize: 0.5, //柱的大小
                minHeight: 0.6,//柱高的最小值
                silent: false, //是否不响应鼠标事件，false为响应，反之为不响应
                itemStyle: {//柱条样式
                    opacity: 1
                },
                // 柱条高亮标签样式
                emphasis: {
                    label: {//开启高亮，但不显示标签，以tooltips代替标签
                        show: false,
                    }
                }
            }]
        };
        // 注册地图到组件, 初始化实例对象
        echarts.registerMap('Shenzhen', regionJson);
        myMap = echarts.init(this.mapRef.current);
        myMap.setOption(option);
        window.onresize = myMap.resize;
    };

    //更新地图
    updateMap = (barData) =>{
        // 如果没有数据则warning
        if (barData.length === 0) {
            // message.warning('No selected data !', 2);//warning
            // 多发生在点击清空按钮后，因此需要将bar图层的数据设置为空
            myMap.setOption({
                series: [{
                    name: '柱子',
                    data: [],
                }]
            });
        } else {//后续这一部分，可以用一个函数来代替，this.XXX(){retuan data}
            // 获取用户id数组
            let data = [];
            for (let i = 0; i < barData.length; i++){
                for (let j = 0; j < barData[i].locations.length; j++){
                    data.push([barData[i].locations[j].lnglat[0], barData[i].locations[j].lnglat[1], barData[i].locations[j].count])
                }
            }

            myMap.setOption({
                series: [{
                    name: '柱子',
                    data: data,
                }]
            });
        }
    };


    componentDidMount() {
        this.initMap();
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        //prevProps获取到的leftWidth是0，在PageSelect页面componentDidMount获取到leftWidth值后，重新初始化
        if(!_.isEqual(prevProps.leftWidth, this.props.leftWidth)||!_.isEqual(prevProps.bottomHeight, this.props.bottomHeight)){
            this.initMap();
        }
        //只要this.context.state.selectedUsers中的值改变，就会在地图上重新渲染
        if(!_.isEqual(prevState.SelectedUsers, this.context.state.selectedUsers)){
            let usersData = this.getUserData(this.context.state.selectedUsers);
            this.updateMap(usersData);
            this.setState({
                SelectedUsers: this.context.state.selectedUsers
            })
        }
    }


    render() {
        return (
            <>
                <div
                    className="select-map-ctn"
                    ref={this.mapRef}
                    >
                </div>
            </>
        );
    }
}

Map.contextType = Store;

export default Map;