import React, { Component } from 'react';
import Charts from '../Charts';
import Calendar from './components/Calendar';
import {Button, Slider} from "antd";
import {RedoOutlined} from "@ant-design/icons";
import {eventEmitter} from "@/common/func/EventEmitter";
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
  getAllUsersData = (minCount, maxCount) => {
      let data = {};
      _.forEach(dateCounts, function (item) {
          let users = [];
          let count = 0;
          for(let i = 0; i < item.userData.length; i++){
              if(item.userData[i].count <= maxCount && item.userData[i].count >= minCount){
                  users.push(item.userData[i].user);
                  count += 1
              }
          }
          data[item.date] = { 'count': count, 'users': users }
      });
      // console.log(data);
      this.data = data;
  };

  getCountRange  = (value) => {
      let minCount = (value[0] <= value[1])? value[0]: value[1];
      let maxCount = (value[0] <= value[1])? value[1]: value[0];
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
      this.getAllUsersData(this.state.minCount, this.state.maxCount);
  }

  componentWillUpdate(nextProps, nextState, nextContext) {
      if(this.state.minCount !== nextState.minCount || this.state.maxCount !== nextState.maxCount){
          this.getAllUsersData(nextState.minCount, nextState.maxCount);
      }
  }


    render() {
    return (
      <><div>
          <Slider
              range
              defaultValue={[0, 2]}
              max={10}
              min={0}
              step={1}
              disabled={false}
              onChange={this.getCountRange}
              onAfterChange = {() =>{
                  let clear = true;
                  eventEmitter.emit('clearCalendarHighlight', {clear});
              }}
          />
          <Button
              ghost
              size='small'
              type='default'
              icon={<RedoOutlined style={{color:'#fff'}}/>}
              onClick={(e) => {
                  let clear = true;
                  eventEmitter.emit('clearCalendarHighlight', {clear})
              }}
              style={{
                  position: 'absolute',
                  right:'10px',
                  zIndex: '9999' //至于顶层
              }}
          /></div>
          <Calendar data={this.data} />
        </>
    );
  }
}

export default ChartRight;