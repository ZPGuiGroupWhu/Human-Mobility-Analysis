import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import 'echarts-wordcloud';
import '../popupContent/PopupContent.scss';

let myChart = null;

export default function WordCloud(props) {
    const { wordData, characterId } = props;
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
        series: [{
            type: 'wordCloud',
            shape: 'square', // 形状
            // maskImage: maskImage, //掩膜形状
            left: 'center',
            top: '15%',
            bottom: '5%',
            width: '100%',
            height: '80%',
            sizeRange: [10, 20], // 字体大小范围
            rotationRange: [0, 0], // 旋转角度
            rotationStep: 0.1,
            gridSize: 10, // 聚集程度
            drawOutOfBound: false,
            textStyle: {
                color: function () {  // 词条颜色根据 人格 来改变
                    switch (characterId) {
                        case 0:
                            return '#FBD786'
                        case 1:
                            return '#f7797d'
                        case 2:
                            return '#6dd5ed'
                        case 3:
                            return '#C6FFDD'
                        default:
                            return 'black'
                    }
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
        if (!ref.current) return () => { };
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


    // 更新词条颜色
    useEffect(() => {
        let color = function () {
            switch (characterId) {
                case 0:
                    return '#FBD786'
                case 1:
                    return '#f7797d'
                case 2:
                    return '#6dd5ed'
                case 3:
                    return '#C6FFDD'
                default:
                    return 'black'
            }
        };
        myChart?.setOption({
            series: [{
                textStyle: {
                    color: color
                }
            }]
        })
    }, [characterId])

    return (
        <div style={{ width: '100%', height: '100%' }}
            ref={ref}
        ></div>
    )
};

