import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import './Histogram.scss';

let myChart = null;

export default function Histogram(props) {
    const ref = useRef(null);

    const { optionData, characterId } = props;

    // 横轴-坐标名称
    const xData = optionData.map(element => Reflect.get(element, 'name'))

    // 纵轴-值
    const yData = optionData.map(element => Reflect.get(element, 'value'))

    // 柱状图参数设置
    const option = {
        // grid - 定位图表在容器中的位置
        grid: {
            show: true, // 是否显示直角坐标系网格
            left: '20', // 距离容器左侧距离
            top: '0', // 距离容器上侧距离
            right: '0',
            bottom: '0',
        },
        // tooltips
        tooltip: {
            trigger: 'item', // 触发类型
            confine: true, // tooltip 限制在图表区域内
            formatter: function (params) {// 说明大五人格
                return params.name + ': ' + params.data;
            },
        },
        // 轴配置
        xAxis: {
            show: true,
            type: 'category',
            data: xData,
            boundaryGap: true,
            nameLocation: 'center',
            // 标签
            axisLabel: {
                show: true,
                interval: 0,
                inside: false, // label显示在外侧
                rotate: 45,
                margin: 5,
                color: "#ccc",
                fontStyle: "normal",
                fontWeight: "bold",
                fontSize: 10
            },
            axisTick: { // 坐标轴刻度线style
                alignWithLabel: true,
                interval: 0,
                inside: false,
                length: 2
            },
            max: xData.length - 1,
        },
        yAxis: {
            show: true,
            type: 'value',
            position: 'left',
            axisLabel: {
                show: true, // 是否显示坐标轴刻度标签
                rotate: 0, // 刻度标签旋转角度
                margin: 2, // 刻度标签与轴线距离
                color: '#ccc', // 刻度标签文字颜色
                fontSize: 10, // 刻度标签文字大小
            },
            splitLine: { // 坐标轴区域的分隔线
                show: true,
                lineStyle: {
                    // 使用深浅的间隔色
                    color: ['#ccc'],
                    width: 0.5
                }
            }, 
            max: 1.05,
            min: -0.01
        },
        series: [
            {
                data: yData,
                type: 'bar',
                showBackground: true, // 是否显示柱条背景颜色
                backgroundStyle: {
                    color: 'rgba(180, 180, 180, 0.2)', // 柱条背景颜色
                    borderColor: '#000', // 描边颜色
                    borderWidth: 0, // 描边宽度
                },
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(
                        0, 0, 0, 1,
                        [
                            { offset: 0, color: '#83bff6' },
                            { offset: 0.5, color: '#188df0' },
                            { offset: 1, color: '#188df0' }
                        ]
                    ), // 柱条颜色
                    borderColor: '#000', // 描边颜色
                    borderWidth: 0, // 描边宽度
                    borderRadius: [5, 5, 0, 0], // 描边弧度
                },
                emphasis: {
                    focus: 'series', // 聚焦效果
                    blurScope: 'coordinateSystem', // 淡出范围
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(
                            0, 0, 0, 1,
                            [
                                { offset: 0, color: '#63b2ee' },
                                { offset: 0.7, color: '#efa666' },
                                { offset: 1, color: '#f89588' }
                            ]
                        )
                    }
                },
            }
        ]
    };
    // 初始化 ECharts 实例对象
    useEffect(() => {
        if (!ref.current) return () => { };
        myChart = echarts.init(ref.current);
        myChart.setOption(option);
        window.onresize = myChart.resize;
    }, [ref]);


    useEffect(() => {
        // 纵轴--值
        const yData = optionData.map(element => Reflect.get(element, 'value'))
        option.series[0].data = yData;
        myChart.setOption(option);
    }, [optionData])


    return (
        <div className='histogram'
            ref={ref}
        ></div>
    )
}
