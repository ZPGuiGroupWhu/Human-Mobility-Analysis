import React, { Component } from "react";
import './TableDrawer.css'
import { Button, Drawer, Table } from 'antd';
import { RightCircleTwoTone, LeftCircleTwoTone } from "@ant-design/icons";
import _ from 'lodash'

export default class TableDrawer extends Component {
  constructor(props) {
    super(props);
    this.Column = [{
      title: '数据',
      dataIndex: 'data',
      key: 'data',
      align: 'center',
    }];
    this.Data = [{
        key: '1',
        data: this.props.radar(),
      }, {
        key: '2',
        data: this.props.wordcloud(),
    },{
        key: '3',
        data: this.props.violinplot(),
      }];
  }

  // 初始化数据
  changeData = () => {
    this.Data = [{
      key: '1',
      data: this.props.radar(),
    }, {
      key: '2',
      data: this.props.wordcloud(),
    }, {
      key: '3',
      data: this.props.violinplot(),
    }]
  };

  //将按钮状态返回给父组件
  toParent = (drawer) => {
    this.props.setDrawerState(drawer);
  };


  componentWillUpdate(nextProps, nextState, nextContext) {
    this.changeData()
  }

  render() {
    return (
      <>
        <Drawer
          closable={false}
          width={this.props.rightWidth + 16}
          keyboard
          mask={false}
          placement='right'
          visible={this.props.rightDrawerVisible}
          bodyStyle={{
            padding: '70px 0px 0px 0px',
            overflowX: 'hidden',
            overflowY: 'hidden'
          }}
        >
          <Table
            showHeader={false}
            scroll={{ y: 'calc(100vh - 70px)'}}
            pagination={false}
            dataSource={this.Data}
            columns={this.Column}
            bordered={true}
            width={this.props.rightWidth}
          />
        </Drawer>
        <Button
          shape="square"
          ghost
          icon={
            this.props.rightBtnChange ?
              <LeftCircleTwoTone twoToneColor="#fff" /> :
              <RightCircleTwoTone twoToneColor="#fff" />
          }
          style={{
            size: 'normal',
            position: 'absolute',
            top: '50%',
            right: (this.props.rightBtnChange ? 0 : this.props.rightWidth + 16) + 'px',
            width: 32,
            transform: 'translateX(0%)',
          }}
          onClick={(e) => {
            this.toParent('right');
          }}
        />
      </>

    );
  }

}