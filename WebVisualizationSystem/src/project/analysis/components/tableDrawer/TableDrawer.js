import React, { Component } from "react";
import './TableDrawer.css'
import { Button, Drawer, Table, Descriptions } from 'antd';
import { RightCircleTwoTone, LeftCircleTwoTone } from "@ant-design/icons";

export default class TableDrawer extends Component {
  constructor(props) {
    super(props);
    this.Column = [{
      title: '数据',
      dataIndex: 'data',
      key: 'data',
      align: 'center',
    }];
    this.leftData = [{
      key: '1',
      data: this.props.radar(),
    }, {
      key: '2',
      data: this.props.wordcloud(),
    }];
    this.rightData = [{
      key: '1',
      data: this.props.violinplot()
    }];

    this.state = {
      leftBtnChange: true,
      leftDrawerVisible: false,
      rightBtnChange: true,
      rightDrawerVisible: false,
    };
  }


  initLeftData = () => {
    this.leftData = [{
      key: '1',
      data: this.props.radar(),
    }, {
      key: '2',
      data: this.props.wordcloud(),
    }];
  };

  changeRightData = () => {
    this.rightData = [{
      key: '1',
      data: this.props.violinplot(),
    }, {
      key: '2',
      data:
        <Descriptions
          column={{ xxl: 4, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }}
          bordered>
          {this.props.data.map(item => (
            <Descriptions.Item
              label={item.option}
              labelStyle={{ textAlign: 'center' }}
              contextStyle={{ textAlign: 'center' }}
            >{item.value.toFixed(5)}</Descriptions.Item>
          ))}
        </Descriptions>
    }]
  };

  //将按钮状态返回给父组件
  toParent = (btn) => {
    this.props.setBtnSate(btn);
  };

  componentWillUpdate(prevProps, prevState, snapshot) {
    this.initLeftData();
    this.changeRightData();
  }

  render() {
    return (
      <>
        <Drawer
          closable={false}
          width={this.props.leftwidth}
          keyboard
          mask={false}
          placement='left'
          visible={this.state.leftDrawerVisible}
          bodyStyle={{
            padding: '70px 0px 0px 0px',
            overflowX: 'hidden',
            overflowY: 'hidden'
          }}
        >
          <Table
            showHeader={false}
            pagination={false}
            dataSource={this.leftData}
            columns={this.Column}
            bordered={true}
          />
        </Drawer>
        <Button
          shape="square"
          ghost
          disabled={this.props.leftBtnDisabled}
          icon={
            this.state.leftBtnChange ?
              <RightCircleTwoTone twoToneColor="#fff" /> :
              <LeftCircleTwoTone twoToneColor="#fff" />
          }
          style={{
            size: 'normal',
            position: 'absolute',
            top: '50%',
            left: (this.state.leftBtnChange ? 0 : this.props.leftwidth) + 'px',
            width: 32,
            transform: 'translateX(0%)',
          }}
          onClick={(e) => {
            this.toParent('leftBtn');
            this.setState(prev => ({
              leftBtnChange: !prev.leftBtnChange,
              leftDrawerVisible: !prev.leftDrawerVisible,
            }))
          }}
        />

        <Drawer
          closable={false}
          width={this.props.rightwidth}
          keyboard
          mask={false}
          placement='right'
          visible={this.state.rightDrawerVisible}
          bodyStyle={{
            padding: '70px 0px 0px 0px',
            overflowX: 'hidden',
            overflowY: 'hidden'
          }}
        >
          <Table
            showHeader={false}
            scroll={{ y: 'calc(100vh - 70px)' }}
            pagination={false}
            dataSource={this.rightData}
            columns={this.Column}
            bordered={true}
          />
        </Drawer>
        <Button
          shape="square"
          ghost
          disabled={this.props.rightBtnDisabled}
          icon={
            this.state.rightBtnChange ?
              <LeftCircleTwoTone twoToneColor="#fff" /> :
              <RightCircleTwoTone twoToneColor="#fff" />
          }
          style={{
            size: 'normal',
            position: 'absolute',
            top: '50%',
            right: (this.state.rightBtnChange ? 0 : this.props.rightwidth) + 'px',
            width: 32,
            transform: 'translateX(0%)',
          }}
          onClick={(e) => {
            this.toParent('rightBtn');
            this.setState(prev => ({
              rightBtnChange: !prev.rightBtnChange,
              rightDrawerVisible: !prev.rightDrawerVisible,
            }))
          }}
        />
      </>

    );
  }

}