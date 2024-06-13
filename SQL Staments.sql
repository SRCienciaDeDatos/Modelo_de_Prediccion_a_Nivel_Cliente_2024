-- Consultas a Eroute


-- ###############                          1) VENTAS HISTÓRICAS DE CLIENTES                     ##################


-- Nota: Vamos a hacer un query del total de ventas historicas en un periodo determinado de tiempo. No importa si los clientes están activos o no, lo que me interesa también es
    -- integrar los patrones de consumo de los clientes por lo que descararé todos los datos ya sea de clientes activos o no

select TransProdDetalle.TransProdID, TransProdDetalle.TransProdDetalleID, TransProd.ClienteClave, ClienteEsquema.EsquemaID, TransProdDetalle.ProductoClave,
       TransProdDetalle.TipoUnidad, TransProdDetalle.Cantidad, TransProdDetalle.Precio, TransProdDetalle.Total,
       TransProdDetalle.MFechaHora, TransProdDetalle.MUsuarioID, Almacen.AlmacenPadreId,  TransProd.Tipo, TransProd.TipoFaseIntSal, TransProd.TipoFase, TransProd.TipoMovimiento
       from TransProdDetalle
       join TransProd on TransProdDetalle.TransProdID = TransProd.TransProdID
       join Almacen on TransProdDetalle.MUsuarioID = Almacen.AlmacenID
       join ClienteEsquema on TransProd.ClienteClave = ClienteEsquema.ClienteClave
        where TransProd.Tipo = 1 and TransProd.TipoFaseIntSal = 1 and TransProd.TipoFase != 0 and TransProd.TipoMovimiento = 2
          and cast(TransProdDetalle.MFechaHora as date)  between '2023-06-14' and '2023-07-14' and Almacen.AlmacenPadreId in ('D001')
        and ClienteEsquema.EsquemaID in ('D21', 'D17', 'D02');
-- Formato es año, mes y día



select top 100 * from ClienteEsquema;


--  ########################                    2) LITROS POR PIEZA                      #########################
select ProductoClave, PRUTipoUnidad, KgLts from  ProductoUnidad;




--  #########################                  3) AGENDA HISTÓRICA                  #####################
select DiaClave, VendedorId, FrecuenciaClave, RUTClave, ClienteClave, ClaveCEDI from AgendaVendedor where ClaveCEDI in ('D001')
and cast(DiaClave as date) between '2023-01-01' and '2023-12-14'





