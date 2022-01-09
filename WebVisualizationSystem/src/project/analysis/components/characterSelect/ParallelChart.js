import React, { useState, useEffect, useRef } from 'react';
import _ from 'lodash';
import './CharacterWindow.scss';
// ECharts
import * as echarts from 'echarts';
// react-redux
import { useDispatch, useSelector } from 'react-redux';
import { setCharacterSelected } from '@/app/slice/analysisSlice'

let myChart = null;

export default function ParallelChart(props) {
    // Echarts 容器实例
    const ref = useRef(null);
    // 获取chart数据
    const {
        data
    } = props;

    // 获取被选择的用户编号
    const userId = data[0].userid;

    const [trajData, setTrajData] = useState([]);
    const dispatch = useDispatch();
    const state = useSelector(state => state.analysis);

    // 特征属性
    const characters = [
        { dim: 0, name: '移动总距离' },
        { dim: 1, name: '速度均值' },
        { dim: 2, name: '转向角均值' }
    ];

    // 计算平均值
    function getAvg(arr) {
        let sum = 0.0;
        arr.forEach((item) => {
            sum += parseFloat(item);
        })
        return sum / arr.length
    }

    // 处理数据
    function handleData(data) {
        const trajData = [];
        _.forEach(data, (item, index) => {
            if (state.finalSelected.includes(item.id)) {
                let avgSpeed = getAvg(item.spd);
                let avgAzimuth = getAvg(item.azimuth);
                let totalDis = item.disTotal;
                trajData.push([totalDis, avgSpeed, avgAzimuth]);
            }
        });
        return trajData;
    }

    // 刷选时更新characterSelected数组
    function onAxisAreaSelected(params) {
        let series0 = myChart.getModel().getSeries()[0];
        let indices0 = series0.getRawIndicesByActiveState('active');
        // console.log(indices0)
        const payload = indices0.map(item => {
            let trajId = [userId, item].join('_'); // 字符串拼接得到轨迹编号
            return trajId;
        });
        dispatch(setCharacterSelected(payload));
    };

    // 选框样式
    const areaSelectStyle = {
        width: 15,
        borderWidth: .8,
        borderColor: 'rgba(160,197,232)',
        color: 'rgba(160,197,232)',
        opacity: .4,
    }

    // 线样式
    const lineStyle = {
        width: 1,
        opacity: 0.5,
        cap: 'round',
        join: 'round',
    };

    const option = {
        // 工具栏配置
        toolbox: {
            iconStyle: {
                color: '#fff', // icon 图形填充颜色
                borderColor: '#fff', // icon 图形描边颜色
            },
            emphasis: {
                iconStyle: {
                    color: '#7cd6cf',
                    borderColor: '#7cd6cf',
                }
            }
        },
        // 框选工具配置
        brush: {
            toolbox: ['clear'],
            xAxisIndex: 0,
            throttleType: 'debounce',
            throttleDelay: 300,
        },
        visualMap: [
            {
                type: 'piecewise',
                min: 0,
                max: 10,
                splitNumber: 6,
                show: false,
                seriesIndex: 0,
                dimension: 0,
                inRange: {
                    color: ['#D00000', '#DC2F02', '#E85D04', '#F48C06', '#FAA307', '#FFBA08']
                }
            }
        ],
        parallelAxis: characters.map((item) => ({
            dim: item.dim,
            name: item.name,
            areaSelectStyle: areaSelectStyle
        })),
        parallel: {
            left: '10%',
            right: '13%',
            bottom: '17%',
            top: '13%',
            parallelAxisDefault: {
                type: 'value',
                nameLocation: 'start',
                nameGap: 20,
                nameTextStyle: {
                    fontSize: 12,
                    color: '#fff',
                },
                min: 'dataMin',
                max: 'dataMax',
                axisLine: {
                    lineStyle: {
                        color: '#fff',
                    }
                },
                axisTick: {
                    show: false,
                },
                axisLabel: {
                    formatter: (value) => (parseInt(value)),
                }
            }
        },
        series: [
            {
                name: '特征筛选',
                type: 'parallel',
                lineStyle: lineStyle,
                data: [],
            },
        ]
    };

    // 初始化 ECharts 实例对象
    useEffect(() => {
        if (!ref.current) return () => {};
        myChart = echarts.init(ref.current);
        myChart.setOption(option);
        myChart.on('axisareaselected', onAxisAreaSelected);
        window.onresize = myChart.resize;
    }, [ref])

    // 当 data改变或者 finalSelected改变时
    useEffect(() => {
        let trajData = handleData(data);
        myChart?.setOption({
            series: [{
                name: '特征筛选',
                data: trajData
            }]
        })
    }, [data, state.finalSelected])

    return (
        <div className='character-parallel'
            ref={ref}
        ></div>
    )
}