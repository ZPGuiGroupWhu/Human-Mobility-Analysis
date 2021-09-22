import React, { Component } from 'react';
import Charts from '../Charts';
import Calendar from './components/Calendar';
import {Button, Slider} from "antd";
import {RedoOutlined} from "@ant-design/icons";
import { eventEmitter } from "@/common/func/EventEmitter";
import _ from 'lodash'
//数据
import dateCounts from './jsonData/date_counts'

const sliderMin = 0;
const sliderMax = 2;

class ChartRight extends Component {
  constructor(props) {
    super(props);
    this.data = {};
    this.state ={
        minCount: sliderMin,
        maxCount: sliderMax,
    };
  }

  // 组织日历数据：{ 日期date：出行用户数量count }
  getDateData = (minCount, maxCount) => {
      let data = {};
      _.forEach(dateCounts, function (item) {
          let users = [];
          let count = 0;//记录该日期下有多少用户
          for (let i = 0; i < item.userData.length; i++) {
              if (item.userData[i].count <= maxCount && item.userData[i].count >= minCount) {
                  users.push(item.userData[i].user);
                  count += 1
              }
          }
          data[item.date] = {'count': count, 'users': users}
      });
      // console.log(data);
      this.data = data;
  };

    getCountRange  = (value) => {
        let minCount = (value[0] <= value[1])? value[0]: value[1];
        let maxCount = (value[0] <= value[1])? value[1]: value[0];
        //清楚高亮标记
        let clear = true;
        eventEmitter.emit('clearCalendarHighlight', {clear});
        //每次设置setState时会重新渲染，因此需要先更新data数据在setState
        this.setState({
            minCount: minCount,
            maxCount: maxCount
        });
    };

  // 加载组件前传递数据给calendar
  componentWillMount() {
      this.getDateData(this.state.minCount, this.state.maxCount);
  }

  componentWillUpdate(nextProps, nextState, nextContext) {
      if(this.state.minCount !== nextState.minCount || this.state.maxCount !== nextState.maxCount){
          this.getDateData(nextState.minCount, nextState.maxCount);
      }
  }

  render(){
      //利用marks可以标注刻度，实现Slider标签及其位置设置
      const marks = {
          0: {
              style: {
                  color: '#fff',
              },
              label: <p style={{
                  position: 'absolute',
                  right: '30px',
                  top: '-20px'
              }}>0</p>
          },
          8: {
              style: {
                  color: '#fff',
              },
              label: <p style={{
                  position: 'absolute',
                  right: '30px',
                  top: '-20px'
              }}>8</p>
          },
      };
    return (
      <>
        <div>
            <Slider
                range
                defaultValue={[0, 2]}
                max={8}
                min={0}
                step={1}
                vertical={true}
                disabled={false}
                tipFormatter={function(value){
                    return '当日出行总数: ' + value;
                }}
                onChange={this.getCountRange}
                onAfterChange={() => {
                    //清楚高亮标记
                    let clear = true;
                    eventEmitter.emit('clearCalendarHighlight', {clear});
                }}
                marks={marks}
                style={{
                    height: '200px',
                    position: 'absolute',
                    right: '5px',
                    top: '100px',
                    zIndex: '9999' //至于顶层
                }}
            />
            <Button
                ghost
                size='small'
                type='default'
                icon={<RedoOutlined style={{color: '#fff'}}/>}
                onClick={(e) => {
                    let clear = true;
                    eventEmitter.emit('clearCalendarHighlight', {clear})
                }}
                style={{
                    position: 'absolute',
                    right: '10px',
                    zIndex: '9999' //至于顶层
                }}
            /></div>
          <Calendar data={this.data}/>
      </>
    );
  }
}

export default ChartRight;