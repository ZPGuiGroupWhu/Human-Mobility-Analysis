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
  toParent = (drawer) => {
    this.props.setDrawerState(drawer);
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
          visible={this.props.leftDrawerVisible}
          bodyStyle={{
            padding: '70px 0px 0px 0px',
            overflowX: 'hidden',
            overflowY: 'hidden'
          }}
        >
          <Table
            showHeader={false}
            // scroll={{ y: 'calc(100vh - 70px)' }}
            pagination={false}
            dataSource={this.leftData}
            columns={this.Column}
            bordered={true}
            width={this.props.leftwidth}
          />
        </Drawer>
        <Button
          shape="square"
          ghost
          icon={
            this.props.leftBtnChange ?
              <RightCircleTwoTone twoToneColor="#fff" /> :
              <LeftCircleTwoTone twoToneColor="#fff" />
          }
          style={{
            size: 'normal',
            position: 'absolute',
            top: '50%',
            left: (this.props.leftBtnChange ? 0 : this.props.leftwidth) + 'px',
            width: 32,
            transform: 'translateX(0%)',
          }}
          onClick={(e) => {
            this.toParent('left');
          }}
        />

        <Drawer
          closable={false}
          width={this.props.rightwidth}
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
            scroll={{ y: 'calc(100vh - 70px)' }}
            pagination={false}
            dataSource={this.rightData}
            columns={this.Column}
            bordered={true}
            width={this.props.rightwidth}
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
            right: (this.props.rightBtnChange ? 0 : this.props.rightwidth) + 'px',
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