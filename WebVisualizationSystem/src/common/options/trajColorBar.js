// 根据距离筛选颜色
export const trajColorBar = [{
  static: '0~5',
  dynamic: 'dt0',
  min: 0,
  max: 5,
  color: '#FDFD00'
}, {
  static: '5~10',
  dynamic: 'dt1',
  min: 5,
  max: 10,
  color: '#9DFD00'
}, {
  static: '10~20',
  dynamic: 'dt2',
  min: 10,
  max: 20,
  color: '#00FDDB'
}, {
  static: '20~30',
  dynamic: 'dt3',
  min: 20,
  max: 30,
  color: '#8600FD'
}, {
  static: '30~40',
  dynamic: 'dt4',
  min: 30,
  max: 40,
  color: '#FD008A'
}, {
  static: '> 40',
  dynamic: 'dt5',
  min: 40,
  max: Infinity,
  color: '#F8F9F9'
}]