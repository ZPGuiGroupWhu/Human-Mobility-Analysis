import React, { Component, createRef } from 'react';
import * as echarts from 'echarts';
import _ from "lodash";
import '../popupContent/PopupContent.scss';

export default class Radar extends Component{
    constructor(props) {
        super(props);
    }
    // ECharts 容器实例
    chartRef = createRef();
    getRadar = () =>{
        const myChart = echarts.init(this.chartRef.current);

        const lineStyle = {
            normal: {
                width: 1.5,
                opacity: 1
            }
        };
        const option = {
            backgroundColor: 'transparent',
            // title: {
            //     text: '大五人格-雷达图',
            //     left: 'left',
            //     textStyle: {
            //         color: '#fff',
            //         fontSize: 12,
            //     }
            // },
            legend: { //图例
                bottom: 0,
                data: ['当前用户', '平均水平'],
                itemGap: 25,
                textStyle: {
                    color: '#fff',
                    fontSize: 10,
                    // fontWeight:'bold',
                },
                // selectedMode: 'single'
            },
            tooltip: { //tooltip显示信息
                trigger: 'item', // 触发类型
                confine: true, // tooltip 限制在图表区域内
                textStyle: {
                    fontSize: 3,
                    fontFamily:'Microsoft Yahei'
                }
            },
            // visualMap: {
            //     show: true,
            //     min: 0,
            //     max: 20,
            //     dimension: 6,
            //     inRange: {
            //         colorLightness: [0.5, 0.8]
            //     }
            // },
            radar: {
                indicator: [
                    {name: '外向性', max: 8},
                    {name: '开放性', max: 8},
                    {name: '神经质性', max: 8},
                    {name: '尽责性', max: 8},
                ],
                radius: 70, // 可视区域大小
                startAngle: 90,
                shape: 'polygon',
                splitNumber: 4,
                center: ['50%', '45%'],
                nameGap: 5,
                name: {
                    textStyle: {
                        color: '#87CEFA',
                        fontSize: 10
                    }
                },
                splitLine: {
                    lineStyle: {
                        color: 'rgba(211, 253, 250, 0.8)'
                    }
                },
                splitArea: {
                    areaStyle: {
                        color: ['#77EADF','#87CEFA', '#64AFE9','#428BD4'],
                        shadowColor: 'rgba(0, 0, 0, 0.2)',
                        shadowBlur: 10
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: 'rgba(211, 253, 250, 0.8)'
                    }
                }
            },
            series: [
                {
                    name: '雷达图',
                    type: 'radar',
                    emphasis: {
                        lineStyle: {
                            width: 5
                        }
                    },
                    data:[{
                        name: '当前用户',
                        lineStyle: lineStyle,
                        value: this.props.radarData[0],
                        itemStyle: {
                            color: '#FFE434'
                        },
                        areaStyle: {
                            opacity: 0.3
                        }
                    }, {
                        name: '平均水平',
                        lineStyle: lineStyle,
                        value: this.props.radarData[1],
                        itemStyle: {
                            color: 	'#67F9D8'
                        },
                        areaStyle: {
                            opacity: 0.1
                        }
                    }]
                }
            ]
        };
        myChart.setOption(option);
        window.onresize = myChart.resize;
    };

    // 初始化
    componentDidMount() {
        this.getRadar();
    }

    componentDidUpdate(prevProps) {
        if(!_.isEqual(this.props.radarData, prevProps.radarData)){
            this.getRadar()
        }
    }

    render() {
        return (
            <div className='radar'
                ref={this.chartRef}
            ></div>
        );
    }
}

