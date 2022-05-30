import React, { useEffect, useState } from "react";
// 样式
import './PopupContent.scss';
import { Select } from 'antd';
// 组件
import Radar from '../radar/Radar'
import WordCloud from '../wordcloud/WordCloud'
import ViolinPlot from "../violinplot/ViolinPlot";
import Description from "../description/Description";
// 大五人格数据
import personalityData from './ocean_score.json';
// 库
import _ from 'lodash';


export default function PopupContent(props) {
    // 接收传来的用户id
    const { id } = props;

    const [option, setOption] = useState('总出行次数') // 初始化select下拉框内的值，后续用于更新
    const [radarData, setRadarData] = useState([]) // 初始化radar数据
    const [wordData, setWordData] = useState([]) // 获取wordcloud数据
    const [violinData, setViolinData] = useState([[], 1]) // 获取violinplot数据和用户在数据中的序号
    const [optionData, setOptionData] = useState([]) // 初始化optionData，用于表示属性表和select

    function getRadarData(userID) {
        const Average = [];
        const Person = [];
        let waiXiangScore = 0;
        let kaiFangScore = 0;
        let shenJingScore = 0;
        let jinZeScore = 0;
        let counts = 0;
        _.forEach(personalityData, function (item) {
            if (item.人员编号.toString() === userID) {
                Person.push(item.外向性.toFixed(3));
                Person.push(item.开放性.toFixed(3));
                Person.push(item.神经质性.toFixed(3));
                Person.push(item.尽责性.toFixed(3));
            }
            waiXiangScore += item.外向性;
            kaiFangScore += item.开放性;
            shenJingScore += item.神经质性;
            jinZeScore += item.尽责性;
            counts += 1;
        });
        //保留9位小数
        Average.push((waiXiangScore /= counts).toFixed(3));
        Average.push((kaiFangScore /= counts).toFixed(3));
        Average.push((shenJingScore /= counts).toFixed(3));
        Average.push((jinZeScore /= counts).toFixed(3));
        return [Person, Average]
    }

    // 获取小提琴数据
    function getOptionData(userID) {
        // 小提琴图，用户数据
        const optionData = [];
        _.forEach(personalityData, function (item) {
            if (item.人员编号.toString() === userID) {
                for (let i = 1; i < Object.keys(item).length - 4; i++) {
                    optionData.push({ 'option': Object.keys(item)[i], 'value': Object.values(item)[i], 'disbale': false })
                }
            }
        });
        return optionData
    }

    // 获取wordcloud数据
    function getWordData(userID) {
        const wordData = [];
        _.forEach(personalityData, function (item) {
            if (item.人员编号.toString() === userID) {
                for (let i = 1; i < Object.keys(item).length - 4; i++) {
                    wordData.push({ 'name': Object.keys(item)[i], 'value': Object.values(item)[i] });
                }
            }
        });
        return wordData
    };

    // 获取用户在数据集中的序号
    function getViolinData(userID) {
        let count = 0;
        let number = 0;
        const violinData = []
        _.forEach(personalityData, function (item) {
            if (item['人员编号'].toString() === userID) {
                number = count;
            }
            violinData.push(item);
            count += 1;
        });
        return [violinData, number]
    }

    // 改变select下拉框的值
    function optionChange(value) {
        // select下拉框改变值
        setOption(value)
    }

    // 组织数据
    useEffect(() => {
        const radarData = getRadarData(id)
        const optionData = getOptionData(id)
        const wordData = getWordData(id)
        const violinData = getViolinData(id)
        console.log(violinData)
        // 更新state
        setRadarData(radarData)
        setOptionData(optionData)
        setWordData(wordData)
        setViolinData(violinData)
    }, [id])

    return (
        <div className="popup-content-ctn">
            <div className="popup-top">
                <>
                    <Radar radarData={radarData} />
                    <WordCloud wordData={wordData} />
                    <div className="clear" />
                </>
            </div>
            <div className="popup-middle">
                <>
                    <div className="select">
                        <Select showSearch={true}
                            defaultValue={'总出行次数'}
                            optionFilterProp="children"
                            notFoundContent="无法找到"
                            onChange={optionChange}
                            style={{
                                width: '100%',
                                fontWeight: 'bold'
                            }}>
                            {optionData.map(item => (
                                <Select.Option key={item.option}>{item.option}</Select.Option>
                            ))}
                        </Select>
                    </div>
                    <ViolinPlot violinData={violinData} option={option}  width={395}/>
                </>
            </div>
            <div className="popup-bottom">
                <Description optionData={optionData} ></Description>
            </div>
            
        </div>
    )
}
