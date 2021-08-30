import React, { Component, createRef } from 'react';
import * as echarts from 'echarts';
import './Radar.css'
import _ from "lodash";

export default class Radar extends Component{
    constructor(props) {
        super(props);
    }
    // ECharts 容器实例
    chartRef = createRef();
    getRadar = () =>{
        const data = this.props.data;
        const id = this.props.id;
        const myChart = echarts.init(this.chartRef.current);
        const Average = [];
        const Person = [];
        let waiXiangScore = 0;
        let kaiFangScore = 0;
        let shenJingScore = 0;
        let jinZeScore = 0;
        let counts = 0;
        _.forEach(data, function(item){
            if (item.人员编号 === id){
                Person.push(item.外向性);
                Person.push(item.开放性);
                Person.push(item.神经质性);
                Person.push(item.尽责性);
            }
            waiXiangScore += item.外向性;
            kaiFangScore += item.开放性;
            shenJingScore += item.神经质性;
            jinZeScore += item.尽责性;
            counts += 1;
        });
        //保留9位小数
        Average.push((waiXiangScore /= counts).toFixed(9));
        Average.push((kaiFangScore /= counts).toFixed(9));
        Average.push((shenJingScore /= counts).toFixed(9));
        Average.push((jinZeScore /= counts).toFixed(9));

        const lineStyle = {
            normal: {
                width: 2,
                opacity: 1
            }
        };
        const option = {
            backgroundColor: 'rgba(250,235,215,0.2)',
            // title: {
            //     text: '大五人格-雷达图',
            //     left: 'center',
            //     textStyle: {
            //         color: '#eee'
            //     }
            // },
            legend: {
                bottom: 5,
                data: ['当前用户', '平均水平'],
                itemGap: 50,
                textStyle: {
                    color: '#696969',
                    fontSize: 12,
                    fontWeight:'bold',
                },
                // selectedMode: 'single'
            },
            tooltip: {
                position: ['10%', '50%'],
                textStyle: {
                    fontSize:5,
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
                radius: 95,
                startAngle: 90,
                shape: 'circle',
                splitNumber: 4,
                center: ['50%', '45%'],
                name: {
                    textStyle: {
                        color: '#428BD4',
                        fontSize: 12
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
                            width: 3
                        }
                    },
                    data:[{
                        name: '当前用户',
                        lineStyle: lineStyle,
                        value: Person,
                        itemStyle: {
                            color: '#67F9D8'
                        },
                        areaStyle: {
                            opacity: 0.3
                        }
                    }, {
                        name: '平均水平',
                        lineStyle: lineStyle,
                        value: Average,
                        itemStyle: {
                            color: 	'#FFE434'
                        },
                        areaStyle: {
                            opacity: 0.1
                        }
                    }]
                }
            ]
        };
        myChart.setOption(option);
    };

    componentDidMount() {
        this.getRadar();
    }

    // // 初始化 ECharts 实例对象
    // useEffect(() => {
    //     if (!ref.current) return () => {};
    //     myChart = echarts.init(ref.current);
    //     myChart.setOption(option);
    //     window.onresize = myChart.resize;
    // }, [ref]);
    render() {
        return (
            <div
                className={'radarChart'}
                ref={this.chartRef}
                style={{
                    width: 300,
                    height: 'calc(50vh - 35px)',
                }}
            ></div>
        );
    }
}

