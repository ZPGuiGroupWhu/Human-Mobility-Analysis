import React, { useEffect, useState } from "react";
// 样式
import './PopupContent.scss';
import { Button, Select } from 'antd';
import { UserSwitchOutlined } from '@ant-design/icons';
// 组件
import Radar from '../radar/Radar'
import WordCloud from '../wordcloud/WordCloud'
import ViolinPlot from "../violinplot/ViolinPlot";
import Description from "../description/Description";
import Histogram from "../histogram/Histogram";
// 大五人格数据
import personalityData from './ocean_score.json';
// 库
import _ from 'lodash';


export default function PopupContent(props) {
    // 接收传来的用户id
    const { id } = props;

    const characterIdMap = {
        '外向': 0,
        '开放': 1,
        '神经质': 2,
        '尽责': 3
    }

    const [characterId, setCharacterId] = useState(0); // 人格属性：外向，开放, 神经质, 尽责
    const [option, setOption] = useState('总出行次数') // 初始化select下拉框内的值，后续用于更新
    const [radarData, setRadarData] = useState([]) // 初始化radar数据
    const [wordData, setWordData] = useState([]) // 获取wordcloud数据
    const [violinData, setViolinData] = useState([[], 1]) // 获取violinplot数据和用户在数据中的序号
    const [optionData, setOptionData] = useState([]) // 初始化optionData，用于表示属性表和select
    const [IsHistogram, setHistogram] = useState(true) // 初始化 popover-bottom放什么数据

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
                for (let groupId = 0; groupId < 4; groupId++) {
                    const groupItem = [];
                    for (let i = groupId * 8 + 1; i <= (groupId + 1) * 8; i++) {
                        groupItem.push({ 'name': Object.keys(item)[i], 'value': Object.values(item)[i] })
                    }
                    wordData.push(groupItem);
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

    function changeBottom() {
        const flag = !IsHistogram;
        setHistogram(flag);
    }

    function characterChange(value) {
        setCharacterId(characterIdMap[value]);
    }

    // 组织数据
    useEffect(() => {
        const radarData = getRadarData(id)
        const optionData = getOptionData(id)
        const wordData = getWordData(id)
        const violinData = getViolinData(id)
        // 更新state
        setRadarData(radarData)
        setOptionData(optionData)
        setWordData(wordData[0])
        setViolinData(violinData)
        setCharacterId(0) // 初始化charaterId
    }, [id])

    // 更新 wordcloud 数据
    useEffect(() => {
        const wordData = getWordData(id);
        setWordData(wordData[characterId])
    }, [characterId])

    return (
        <div className="popup-content-ctn">
            <div className="popup-top">
                <>
                    <div className="title-bar">
                        <div className="character-title">{'大五人格'}</div>
                        <div className="character-select">
                            <Select
                                defaultValue={'外向'}
                                optionFilterProp="children"
                                size="small"
                                onChange={characterChange}
                                style={{
                                    width: '75px',
                                    fontSize: '10px',
                                    marginTop: '5px',
                                    marginRight: '5px',
                                    padding: '0px',
                                    border: '1.5px solid rgb(187, 255, 255)',
                                    borderRadius: '5px',
                                }}
                            >
                                <Select.Option value="外向" style={{
                                    fontSize: '10px',
                                }}>外向</Select.Option>
                                <Select.Option value="开放" style={{
                                    fontSize: '10px',
                                }}>开放</Select.Option>
                                <Select.Option value="神经质" style={{
                                    fontSize: '10px',
                                }}>神经质</Select.Option>
                                <Select.Option value="尽责" style={{
                                    fontSize: '10px',
                                }}>尽责</Select.Option>
                            </Select>
                        </div>
                    </div>
                    <Radar radarData={radarData} />
                    <div className="wordcloud">
                        {/* <WordCloud wordData={wordData} characterId={characterId} /> */}
                        <Histogram optionData={wordData} ></Histogram>
                    </div>
                    
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
                                fontWeight: 'bold',
                                backgroundColor: 'black'
                            }}>
                            {optionData.map(item => (
                                <Select.Option key={item.option}>{item.option}</Select.Option>
                            ))}
                        </Select>
                    </div>
                    <ViolinPlot violinData={violinData} option={option} width={395} />
                </>
            </div>
            <div className="popup-bottom">
                {/* {IsHistogram ?
                    <Histogram optionData={wordData} ></Histogram> :
                    <Description optionData={wordData}></Description>
                } */}
            </div>
            <Button className="switch"
                type='ghost'
                icon={<UserSwitchOutlined />}
                shape='default'
                style={{
                    height: '15px',
                    width: '15px'
                }}
                onClick={changeBottom}
            />
        </div>
    )
}
