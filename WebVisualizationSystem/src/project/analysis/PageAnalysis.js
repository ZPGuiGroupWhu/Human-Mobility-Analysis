import React, { Component } from 'react';
import { Select, Button, Descriptions } from 'antd';
import { withRouter } from 'react-router';
// 样式
import '../bmap.scss';
// 伪数据
import userData from './components/deckGL/399313.json'
import personalityData from './components/tableDrawer/radar/ocean_score.json'
// 自定义组件
import DeckGLMap from './components/deckGL/DeckGLMap';
import CalendarDrawer from './components/calendar/CalendarDrawer';
import Calendar from './components/calendar/Calendar';
import TableDrawer from "./components/tableDrawer/TableDrawer";
import Radar from "./components/tableDrawer/radar/Radar";
import WordCloud from "./components/tableDrawer/wordcloud/WordCloud";
import ViolinPlot from "./components/tableDrawer/violinplot/ViolinPlot";
import _ from "lodash";
import { BlockOutlined } from "@ant-design/icons";


/**
 * props
 * @param {number[] | string} initCenter - 初始中心经纬度 | 城市名（如‘深圳市’）
 * @param {number} initZoom - 初始缩放级别
 */

//某一个用户的id，之后应当放在类里面用于接收传来的用户id
const userID = 100045440;
// 小提琴图的初始label
const initlabel = '总出行次数';
// 小提琴图，用户数据
const optionData = [];
_.forEach(personalityData, function (item) {
  if (item.人员编号 === userID) {
    for (let i = 1; i < Object.keys(item).length - 4; i++) {
      optionData.push({ 'option': Object.keys(item)[i], 'value': Object.values(item)[i], 'disbale': false })
    }
  }
});

class PageAnalysis extends Component {
  constructor(props) {
    super(props);
    this.EVENTNAME = 'showTraj';
    // 词云掩码
    this.maskImage = new Image();
    this.maskImage.src = "data:image/png;base64,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"
    // date：筛选的日期、option：筛选的题项
    this.state = {
      rightWidth: 380,
      date: null,
      flag: true,
      option: initlabel,
      rightBtnChange: true,
      rightDrawerVisible: false,
      bottomDrawerVisible: false,
    }
  };

  getTrajCounts = (count) => {
    this.setState({
      date: count,
    })
  };

  setDrawerState = (drawer) => {
    if (drawer === 'right') {
      this.setState(prev => ({
        rightBtnChange: !prev.rightBtnChange,
        rightDrawerVisible: !prev.rightDrawerVisible,
        bottomDrawerVisible: false
      }))
    }
    if (drawer === 'bottom') {
      this.setState(prev => ({
        bottomDrawerVisible: !prev.bottomDrawerVisible,
        rightBtnChange: true,
        rightDrawerVisible: false
      }))
    }
  };

  // select下拉框改变值
  optionChange = (value) => {
    // console.log(value);
    this.setState({
      option: value
    })
  };
  // Button按钮切换flag的值，从而确定显示词云图还是数据表格
  switchData = (event) => {
    this.setState({
      flag: !this.state.flag
    })
  };

  render() {
    return (
      <>
        <DeckGLMap
          userData={userData}
          getTrajCounts={this.getTrajCounts}
          eventName={this.EVENTNAME}
          setRoutes={this.props.setRoutes}
        />
        <TableDrawer radar={() => (
          <div>
            <Radar data={personalityData} eventName={this.EVENTNAME} id={100045440} rightWidth={this.state.rightWidth} />
          </div>)}
          wordcloud={() => (
            //根据flag判断，切换为词云图还是数据表格
            this.state.flag ?
              <div>
                {/*按钮属性*/}
                <Button shape="square"
                  ghost
                  size='small'
                  style={{
                    position: 'absolute',
                    top: 5,
                    right: 2,
                    zIndex: 1,
                    borderWidth: 1.5,
                    color: '#D2691E',
                    borderColor: '#D2691E',
                  }}
                  icon={
                    <BlockOutlined />
                  }
                  onClick={this.switchData} />
                <WordCloud data={personalityData} eventName={this.EVENTNAME} id={100045440}
                  maskImage={this.maskImage} rightWidth={this.state.rightWidth} />
              </div> :
              <div>
                <Button shape="square"
                  ghost
                  size='small'
                  style={{
                    position: 'absolute',
                    top: 5,
                    right: 2,
                    zIndex: 1,
                    borderWidth: 1.5,
                    color: '#D2691E',
                    borderColor: '#D2691E',
                  }}
                  icon={
                    <BlockOutlined />
                  }
                  onClick={this.switchData} />
                <Descriptions
                  title='用户属性表'
                  column={{ xxl: 4, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }}
                  bordered>
                  {optionData.map(item => (
                    <Descriptions.Item
                      label={item.option}
                      labelStyle={{ textAlign: 'center' }}
                      contextStyle={{ textAlign: 'center' }}
                    >{item.value.toFixed(5)}</Descriptions.Item>
                  ))}
                </Descriptions>
              </div>)}
          violinplot={() => (
            <div>
              <p></p>
              <Select showSearch={true}
                style={{ width: this.state.rightWidth - 30 }}
                defaultValue={initlabel}
                optionFilterProp="children"
                notFoundContent="无法找到"
                onChange={this.optionChange}>
                {optionData.map(item => (
                  <Select.Option key={item.option}>{item.option}</Select.Option>
                ))}
              </Select>
              <ViolinPlot data={personalityData} eventName={this.EVENTNAME} id={100045440}
                option={this.state.option} rightWidth={this.state.rightWidth} />
            </div>)}
          rightWidth={this.state.rightWidth} data={optionData} eventName={this.EVENTNAME}
          rightDrawerVisible={this.state.rightDrawerVisible}
          rightBtnChange={this.state.rightBtnChange}
          setDrawerState={this.setDrawerState} />
        <CalendarDrawer render={() => (<Calendar data={this.state.date} eventName={this.EVENTNAME} />)}
          height={170}
          bottomDrawerVisible={this.state.bottomDrawerVisible}
          setDrawerState={this.setDrawerState} />
        {/* <Button
          onClick={() => {
            this.props.history.push('/select/predict')
          }}
          style={{ position: 'fixed', bottom: '20px', left: '20px', zIndex: '9999' }}
        >
          预测
        </Button> */}
      </>
    )
  }
}

export default withRouter(PageAnalysis)