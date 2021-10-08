import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import _ from "lodash";

export default class ViolinPlot extends Component{
    constructor(props) {
        super(props);
        // 用于记录该id的用户在数据集内的序号
        this.number = 0;
    };

    chartRef = createRef();
    unPack(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }

    getViolinPlot = () => {
        const id = this.props.id;
        const data = this.props.data;
        const option = this.props.option;
        const userData = [];
        let count = 0;
        let number = 0;
        _.forEach(data, function(item){
            if(item['人员编号'] === id){
                number = count;
            }
            userData.push(item);
            count += 1;
        });

        const drawData = [{
            type: 'violin',
            x: this.unPack(userData, option),
            hoverinfo: 'x',
            hoverlabel:{
                bgcolor:'#FFC0CB',
                bordercolor: 'grey',
                font:{
                    family:'Microsoft Yahei',
                    size: 10,
                    color: 'black'
                }
            },
            hoveron: 'violins+kde',
            meanline:{
                visable: true,
                width: 3,
                color:'#FF8C00'
            },
            box: {
                visible: true,
                width: 0.05,
                // fillcolor:'black',
                line:{
                    color: 'black',
                    width: 2
                }
            },
            spanmode: 'soft',
            line: {
                color: 'black',
                width: 1.5
            },
            fillcolor: '#87CEEB',
            opacity: 0.6,
            marker:{
                symbol:'diamond',
            },
            points: 'all',
            pointpos: 0,
            selectedpoints: [number],
            selected:{
                marker:{
                    color: '#FF0000',
                    size: 6,
                }
            },
            unselected:{
                marker:{
                    opacity:0,
                }
            },
            y0: "Total Bill"
        }];

        const layout = {
            paper_bgcolor: 'rgba(250,235,215,0.1)',
            plot_bgcolor: 'rgba(250,235,215,0.1)',
            xaxis: {
                zeroline: false,
            },
            yaxis: {
                showline: true
            },
            height: 200,
            width: this.props.rightWidth,
            margin:{
                b:20,
                l:0,
                r:0,
                t:10
            }
        };

        //加载图表
        Plotly.newPlot(this.chartRef.current, drawData, layout);
        // resize
        window.onresize = Plotly.resize;
    };
    componentDidMount() {
        this.getViolinPlot();
    }
    componentDidUpdate(prevProps, prevState, snapshot) {
        if(!_.isEqual(prevProps.option, this.props.option)){
            console.log(this.props.option);
            this.getViolinPlot();
        }
    }

    render() {
        return (
            <div
                className={'violinPlot'}
                ref={this.chartRef}
            ></div>
        );
    }
}



