import React, {Component} from "react";
import './TableDrawer.css'
import {Button, Drawer, Table} from 'antd';
import {UpCircleOutlined , DownCircleOutlined} from "@ant-design/icons";

export default class TableDrawer extends Component{
    constructor(props) {
        super(props);
        this.Column = [{
            title: '大五人格雷达图',
            dataIndex: 'radar',
            key: 'radar',
            align: 'center',
        }, {
            title: '用户特征词云图',
            dataIndex: 'wordcloud',
            key: 'wordcloud',
            align: 'center',
        },{
            title: '用户特征小提琴图',
            dataIndex: 'violin',
            key: 'violin',
            align: 'center',
        }];
        this.Data = [{
            key: '1',
            radar: this.props.radar(),
            wordcloud: this.props.wordcloud(),
            violin: this.props.violinplot()
        }];
        this.state = {
            btnChange: true,
            drawerVisible: false,
        };
    }

    changeData = () =>{
        this.Data = [{
            key: '1',
            radar: this.props.radar(),
            wordcloud: this.props.wordcloud(),
            violin: this.props.violinplot()
        }];
    };

    componentWillUpdate(prevProps, prevState, snapshot) {
        this.changeData();
    }

    render() {
        return (
            <>
                <Drawer
                    closable={false}
                    height={this.props.height}
                    keyboard
                    mask={false}
                    placement='top'
                    visible={this.state.drawerVisible}
                    bodyStyle={{
                        padding: '0px 0px 0px 0px',
                    }}
                >
                    <Table
                        showHeader={true}
                        shape={'circle'}
                        pagination={false}
                        dataSource={this.Data}
                        columns={this.Column}
                        bordered={true}
                    />
                </Drawer>
                <Button
                    shape="square"
                    ghost
                    icon={
                        this.state.btnChange ?
                            <DownCircleOutlined twoToneColor="#fff" /> :
                            <UpCircleOutlined twoToneColor="#fff" />
                    }
                    style={{
                        size:'small',
                        position: 'absolute',
                        top: (this.state.btnChange ? 0 : this.props.height-75) + 'px',
                        left: '96%',
                        // width:90,
                        transform: 'translateX(-50%)',
                    }}
                    onClick={(e) => {
                        this.setState(prev => ({
                            btnChange: !prev.btnChange,
                            drawerVisible: !prev.drawerVisible,
                        }))
                    }}
                >统计信息</Button>
            </>

        );
    }

}