import React, { useEffect, useState, useRef } from "react";
// 样式
import './PopupContent.scss';
import { Button, Select, Tooltip } from 'antd';
import { UserSwitchOutlined } from '@ant-design/icons';
// 组件
import Radar from '../radar/Radar'
import WordCloud from '../wordcloud/WordCloud'
import ViolinPlot from "../violinplot/ViolinPlot";
import Description from "../description/Description";
import Histogram from "../histogram/Histogram";
import Pie from "../pie/Pie";
// 库
import _ from 'lodash';
// react redux
import { useSelector } from "react-redux";

export default function PopupContent(props) {
    // 接收传来的用户id
    const { id } = props;

    const characterIdMap = {
        '外向': 0,
        '开放': 1,
        '神经质': 2,
        '尽责': 3
    }

    const select = useSelector(state => state.select);

    const [characterId, setCharacterId] = useState(0); // 人格属性：外向，开放, 神经质, 尽责
    const [option, setOption] = useState('总出行次数') // 初始化select下拉框内的值，后续用于更新
    const [radarData, setRadarData] = useState([]) // 初始化radar数据
    const [wordData, setWordData] = useState([]) // 获取wordcloud数据
    const [violinData, setViolinData] = useState([[], 1]) // 获取violinplot数据和用户在数据中的序号
    const [optionData, setOptionData] = useState([]) // 初始化optionData，用于表示属性表和select
    const [isWordCloud, setWordCloud] = useState(false) // 初始化 右上角是放词云还是玫瑰图
    const [isFold, setFold] = useState(true) // 属性表是否折叠
    const [isFirst, setIsFirst] = useState(false) // 第一次渲染标记

    function getRadarData(userID) {
        const Average = [];
        const Person = [];
        let waiXiangScore = 0;
        let kaiFangScore = 0;
        let shenJingScore = 0;
        let jinZeScore = 0;
        let counts = 0;
        _.forEach(select.OceanScoreAll, function (item) {
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
        _.forEach(select.OceanScoreAll, function (item) {
            if (item.人员编号.toString() === userID) {
                for (let i = 1; i < Object.keys(item).length - 4; i++) {
                    optionData.push({ 'name': Object.keys(item)[i], 'value': Object.values(item)[i], 'disbale': false })
                }
            }
        });
        return optionData
    }

    // 获取wordcloud数据
    function getWordData(userID) {
        const wordData = [];
        _.forEach(select.OceanScoreAll, function (item) {
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
        _.forEach(select.OceanScoreAll, function (item) {
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

    // 切换词云 / 玫瑰图
    function changeTopRightPart() {
        const flag = !isWordCloud;
        setWordCloud(flag);
    }

    // 切换大五人格
    function characterChange(value) {
        setCharacterId(characterIdMap[value]);
    }

    // 点击展开/收起
    function footerClick(e) {
        setFold(prev => (!prev));
    }

    // 请求数据后重新渲染
    useEffect(() => {
        // 数据未请求成功
        if (select.OceanReqStatus !== 'succeeded') return;
        // 初次渲染
        if (select.OceanReqStatus === 'succeeded' && !isFirst) {
            setIsFirst(true);
            const radar = getRadarData(id)
            const option = getOptionData(id)
            const word = getWordData(id)
            const violin = getViolinData(id)
            // 更新state
            setRadarData(radar)
            setOptionData(option)
            setWordData(word[0])
            setViolinData(violin)
            setCharacterId(0) // 初始化 charaterId
        }
    }, [select.OceanReqStatus])

    // 组织数据
    useEffect(() => {
        const radar = getRadarData(id)
        const option = getOptionData(id)
        const word = getWordData(id)
        const violin = getViolinData(id)
        // 更新state
        setRadarData(radar)
        setOptionData(option)
        setWordData(word[0])
        setViolinData(violin)
        setCharacterId(0) // 初始化 charaterId
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
                        <Tooltip placement="topRight"
                            title={isWordCloud ? '切换玫瑰图' : '切换词云图'}
                            color="#ffc0cb"
                        >
                            <Button className="switch"
                                type='ghost'
                                icon={<UserSwitchOutlined />}
                                shape='default'
                                style={{
                                    height: '15px',
                                    width: '15px'
                                }}
                                onClick={changeTopRightPart}
                            /></Tooltip>
                        <div className="character-select">
                            <Select
                                defaultValue={'外向'}
                                optionFilterProp="children"
                                size="small"
                                onChange={characterChange}
                                style={{
                                    width: '65px',
                                    fontSize: '5px',
                                    marginTop: '4px',
                                    marginRight: '3px',
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
                        {
                            isWordCloud ?
                                <WordCloud wordData={wordData} characterId={characterId} /> :
                                <Pie wordData={wordData}></Pie>
                        }
                    </div>
                    <div className="clear" />
                </>
            </div>
            <div className="popup-middle">
                <div className="select">
                    <Select showSearch={true}
                        defaultValue={'总出行次数'}
                        optionFilterProp="children"
                        notFoundContent="无法找到"
                        onChange={optionChange}
                        style={{
                            width: '100%',
                            fontWeight: 'bold',
                        }}>
                        {optionData.map(item => (
                            <Select.Option key={item.name}>{item.name}</Select.Option>
                        ))}
                    </Select>
                </div>
                <ViolinPlot violinData={violinData} option={option} width={395} />
            </div>
            <div className="popup-bottom"
                style={{
                    maxHeight: isFold ? '0px' : '125px',
                    opacity: isFold ? '0' : '1',
                    borderWidth: isFold ? '0px' : '1px'
                }}
            >
                <Description optionData={optionData}></Description>
                {/* {IsHistogram ?
                    <Histogram optionData={wordData} ></Histogram> :
                    <Description optionData={wordData}></Description>
                } */}
            </div>
            <footer className="popup-fold" onClick={footerClick}>{isFold ? '点击展开⬇' : '点击收起⬆'}</footer>
        </div>
    )
}
