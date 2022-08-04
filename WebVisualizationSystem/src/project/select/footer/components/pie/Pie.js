import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
// import './Pie.scss';

let myChart = null;

export default function Pie(props) {
    const ref = useRef(null);

    const { wordData } = props;
    
    // 饼图参数设置
    const option = {
        legend: { // 图例
            show: false,
            top: 'bottom'
        },
        toolbox: {
            show: false,
            feature: {
                mark: { show: true },
                dataView: { show: true, readOnly: false },
                restore: { show: true },
                saveAsImage: { show: true }
            }
        },
        // tooltips
        tooltip: {
            trigger: 'item', // 触发类型
            confine: true, // tooltip 限制在图表区域内
            formatter: function (params) {// 说明大五人格
                return params.name + ': ' + params.data.value;
            },
            textStyle: {
                fontSize: 3,
                fontFamily:'Microsoft Yahei'
            }
        },
        series: [
            {
                name: 'Pie',
                type: 'pie',
                radius: [0, '90%'],
                top: 35,
                center: ['50%', '50%'],
                roseType: 'area',
                stillShowZeroSum: true, // 0值是否显示
                avoidLabelOverlap: true, // 防止标签重叠策略
                percentPrecision: 2, // 百分比精度，2位小数
                itemStyle: {
                    borderRadius: 5
                },
                label: {
                    show: true,
                    color: '#fff',
                    position: 'outside', // 引导线 + 标签
                    fontSize: 7,
                    backgroundColor: 'transparent',
                    overflow: "truncate",  // 文字截断
                    ellipsis: '...',  // 截断后的文字用 ... 表示
                    alignTo: "labelLine",
                },
                data: [],
            }]

    };
    // 初始化 ECharts 实例对象
    useEffect(() => {
        if (!ref.current) return () => { };
        myChart = echarts.init(ref.current);
        myChart.setOption(option);
        window.onresize = myChart.resize;
    }, [ref]);


    useEffect(() => {
        option.series[0].data = wordData;
        myChart.setOption(option);
    }, [wordData])


    return (
        <div style={{width: '100%', height: '100%'}}
            ref={ref}
        ></div>
    )
}
