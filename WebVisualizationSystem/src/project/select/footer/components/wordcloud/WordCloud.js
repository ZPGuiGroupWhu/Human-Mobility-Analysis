import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import 'echarts-wordcloud';
import '../popupContent/PopupContent.scss';

let myChart = null;

export default function WordCloud(props){
    const { wordData } = props;
    // ECharts 容器实例
    const ref = useRef(null);

    // 词云图属性
    const option = {
        backgroundColor: 'transparent',
        tooltip: { //tooltip属性
            show: true,
            trigger: 'item', // 触发类型
            confine: true, // tooltip 限制在图表区域内
            textStyle: {
                fontSize: 2,
                fontFamily: 'Microsoft Yahei'
            }
        },
        series: [ {
            type: 'wordCloud',
            shape: 'circle', // 形状
            // maskImage: maskImage, //掩膜形状
            left: 'center',
            top: 'center',
            right: 'center',
            bottom: 'center',
            width: '100%',
            height: '100%',
            sizeRange: [5, 12], // 字体大小范围
            rotationRange: [-90, 90], // 旋转角度
            rotationStep: 45,
            gridSize: 3, // 聚集程度
            drawOutOfBound: false,
            textStyle: {
                color: function () {
                    return 'rgb(' + [
                        Math.round(Math.random() * 255),
                        Math.round(Math.random() * 200),
                        Math.round(Math.random() * 255)
                    ].join(',') + ')';
                }
            },
            emphasis: {
                textStyle: {
                    shadowBlur: 10,
                    shadowColor: '#fff'
                }
            },
            // Data is an array. Each array item must have name and value property.
            data: [],
        }]
    };

    // 初始化 ECharts 实例对象
    useEffect(() => {
        if (!ref.current) return () => {};
        myChart = echarts.init(ref.current);
        myChart.setOption(option);
        window.onresize = myChart.resize;
    }, [ref]);

    useEffect(() => {
        myChart?.setOption({
            series: [{
                data: wordData
            }]
        });
    }, [wordData])

    return(
        <div className='wordcloud'
            ref={ref}
        ></div>
    )
};

