import React, { Component } from 'react';
import Calendar from './components/Calendar';
import { Button, Slider, Tooltip } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { eventEmitter } from "@/common/func/EventEmitter";
import _ from 'lodash'
//数据
import dateCounts from './jsonData/date_counts';
// react-redux
import { connect } from 'react-redux';
import { setSelectedBySlider } from '@/app/slice/selectSlice';

const sliderMin = 0;
const sliderMax = 8;

class ChartBottom extends Component {
  constructor(props) {
    super(props);
    this.data = {};
    this.allUsers = []; // 存放所有用户编号， 用于初始化数据
    this.state = {
      minCount: sliderMin,
      maxCount: sliderMax,
      calendarReload: {}
    };
  }

  // 组织日历数据：{ 日期date：出行用户数量count }
  getDateData = (minCount, maxCount) => {
    let data = {};
    let allUsers = [];
    _.forEach(dateCounts, function (item) {
      let users = [];
      let count = 0;//记录该日期下有多少用户
      for (let i = 0; i < item.userData.length; i++) {
        if (item.userData[i].count <= maxCount && item.userData[i].count >= minCount) {
          users.push(item.userData[i].user);
          count += 1;
          if(!allUsers.includes(parseInt(item.userData[i].user))){
            allUsers.push(parseInt(item.userData[i].user));
          }
        }
      }
      data[item.date] = { 'count': count, 'users': users }
    });
    // console.log(data);
    this.data = data;
    this.allUsers = allUsers;
  };

  // 根据 selectedUsers 数组 重新组织日历数据
  reloadDateData = (selectedUsers, minCount, maxCount) => {
    let data = {};
    _.forEach(dateCounts, function (item) {
      let users = [];
      let count = 0;//记录该日期下有多少用户
      for (let i = 0; i < item.userData.length; i++) {
        if (item.userData[i].count <= maxCount && item.userData[i].count >= minCount) {
          if(selectedUsers.includes(parseInt(item.userData[i].user))){
            users.push(item.userData[i].user);
            count += 1
          }
        }
      }
      data[item.date] = { 'count': count, 'users': users }
    });
    // console.log(data);
    this.data = data;
  };

  getCountRange = (value) => {
    let minCount = (value[0] <= value[1]) ? value[0] : value[1];
    let maxCount = (value[0] <= value[1]) ? value[1] : value[0];
    //每次设置setState时会重新渲染，因此需要先更新data数据在setState
    this.setState({
      minCount: minCount,
      maxCount: maxCount
    });
  };

  // slider滑动回调函数 返回日期
  onSliderAfterChange = async (value) => {
    this.getCountRange(value);
    this.setState({
      calendarReload: {}
    })
  }

  // 加载组件前传递数据给calendar
  componentWillMount() {
    this.getDateData(this.state.minCount, this.state.maxCount);
  }

  // 重新加载数据 以实现日历的重新渲染
  componentWillUpdate(nextProps, nextState, nextContext) {
    if (this.state.minCount !== nextState.minCount || this.state.maxCount !== nextState.maxCount) {
      this.reloadDateData(this.props.selectedUsers, nextState.minCount, nextState.maxCount);
    }
    if(!_.isEqual(nextProps.selectedUsers, this.props.selectedUsers)){
      this.reloadDateData(nextProps.selectedUsers, this.state.minCount, this.state.maxCount)
    }
  }

  render() {
    //计算slider高度, 和chart的visualMap一样高
    const sliderHeight = this.props.bottomHeight - 50;
    //利用marks可以标注刻度，实现Slider标签及其位置设置
    const marks = {
      0: {
        style: {
          color: '#fff',
        },
        label: <p style={{
          position: 'absolute',
          right: '20px',
          top: '-18px',
        }}>0</p>
      },
      8: {
        style: {
          color: '#fff',
        },
        label: <p style={{
          position: 'absolute',
          right: '20px',
          top: '-18px',
        }}>8</p>
      },
    };
    return (
      <>
        <div>
          <Slider
            range
            defaultValue={[0, 8]}
            max={8}
            min={0}
            step={1}
            vertical={true}
            disabled={false}
            tipFormatter={function (value) {
              return '当日出行总数: ' + value;
            }}
            onAfterChange={(value) => {
              this.onSliderAfterChange(value);
            }}
            marks={marks}
            style={{
              display: 'inline-block',
              height: sliderHeight,
              position: 'absolute',
              left: this.props.bottomWidth - 35,
              top: 35,
              // bottom: -15,
              zIndex: '2' //至于顶层
            }}
          />
          <Tooltip title="还原">
            <Button
              ghost
              disabled={false}
              icon={<ReloadOutlined />}
              size={'small'}
              onClick={() => {
                // calendarReload标记，用于后续清除selectedByCalendar数据
                this.setState({
                  calendarReload: {}
                })
              }}
              style={{
                display: 'inline-block',
                position: 'absolute',
                left: this.props.bottomWidth - 30,
                top: 5,
                zIndex: '2' //至于顶层
              }}
            />
          </Tooltip>
        </div>
        <Calendar
          data={this.data}
          bottomHeight={this.props.bottomHeight}
          bottomWidth={this.props.bottomWidth + 10}
          allUsers={this.allUsers}
          calendarReload={this.state.calendarReload}
        />
      </>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    selectedUsers: state.select.selectedUsers,
    selectedBySlider:  state.select.setSelectedBySlider
  }
}

const mapDispatchToProps = (dispatch) => {
  return {
    setSelectedBySlider: (payload) => dispatch(setSelectedBySlider(payload))
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(ChartBottom);