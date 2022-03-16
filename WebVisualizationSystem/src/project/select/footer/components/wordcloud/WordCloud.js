import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import 'echarts-wordcloud';
import _ from "lodash";
import '../popupContent/PopupContent.scss';

let myChart = null;

export default function WordCloud(props){
    const { wordData } = props;
    // ECharts 容器实例
    const ref = useRef(null);

    // 词云图属性
    const option = {
        backgroundColor:'black',
        tooltip: { //tooltip属性
            show: true,
            position: function (pos, params, dom, rect, size) {
                // 鼠标在左侧时 tooltip 显示到右侧，鼠标在右侧时 tooltip 显示到左侧。
                let obj = {};
                obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 10;
                obj[['bottom','top'][+(pos[1] < size.viewSize[1] / 2)]] = 5;
                return obj;
            },
            textStyle: {
                fontSize: 2,
                fontFamily: 'Microsoft Yahei'
            }
        },
        series: [ {
            type: 'wordCloud',
            name: 'wordcloud',
            shape: 'circle',
            gridSize: 5,//字体分布的密集程度
            // maskImage: maskImage, //掩膜形状
            left: 'center',
            top: 'center',
            right: 'center',
            bottom: 'center',
            width: '100%',
            height: '100%',
            sizeRange: [5, 12],

            // Text rotation range and step in degree. Text will be rotated randomly in range [-90, 90] by rotationStep 45

            rotationRange: [-90, 90],
            rotationStep: 45,

            // size of the grid in pixels for marking the availability of the canvas
            // the larger the grid size, the bigger the gap between words.

            gridSize: 1,

            // set to true to allow word being draw partly outside of the canvas.
            // Allow word bigger than the size of the canvas to be drawn
            drawOutOfBound: false,

            // If perform layout animation.
            // NOTE disable it will lead to UI blocking when there is lots of words.
            layoutAnimation: true,
            textStyle: {
                color: function () {
                    return 'rgb(' + [
                        Math.round(Math.random() * 160),
                        Math.round(Math.random() * 160),
                        Math.round(Math.random() * 160)
                    ].join(',') + ')';
                }
            },
            emphasis: {
                textStyle: {
                    shadowBlur: 10,
                    shadowColor: '#333'
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
                name: 'wordcloud',
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

