
def sql_quary_target():
    return f'''
        select c."name", t.transport_name, t.vehicle_passport_number , tt."name"  ,teot.start_datetime , 
        teot.end_datetime , teot.in_movement , teot.without_movement 
        from public.transport t 
        join public.sensor s on t.tracker_id =s.id 
        join public.construction c on t.construction_id =c.id 
        join public.transport_type tt on tt.id =t.type_transport_id 
        join public.transport_engine_on_time teot on teot.sensor_id =s.id 
    '''
